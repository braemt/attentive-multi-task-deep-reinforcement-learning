#!/usr/bin/env python
import cv2
import go_vncdriver
import os
import tensorflow as tf
import argparse
import logging
import sys, signal
import time
from a3c import A3C
from envs import create_env

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    save_path = None

    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True, write_state=True):
        if self.save_path is None:
            super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                        meta_graph_suffix, False, write_state)
        else:
            super(FastSaver, self).save(sess, self.save_path, global_step, latest_filename,
                                        meta_graph_suffix, False, write_state)


def run(args, server):
    env_ids = str(args.env_id).split(",")
    tasks = len(env_ids)
    original_logdir = args.log_dir
    logdir = os.path.join(args.log_dir, 'train')

    env_task = args.task % tasks
    args.env_id = env_ids[env_task]
    env = create_env(args.env_id)
    ac_spaces = [create_env(env_id).action_space.n for env_id in env_ids]
    workers_per_task = args.num_workers / tasks
    trainer = A3C(env, args.task, env_task, tasks, ac_spaces, workers_per_task)

    trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def global_var(var):
        if not var.name.startswith("global"):
            return False

        if var.name.startswith("global/global_step"):
            return True

        if "/Adam" in var.name:
            return False

        for v in trainable_var_list:
            if v.name == var.name:
                return True

    local_var_list = [v for v in tf.global_variables() if not global_var(v)]
    global_var_list = [v for v in tf.global_variables() if global_var(v)]

    logger.info('Global vars:')
    for v in global_var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    logger.info('Local vars:')
    for v in local_var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    logger.info('Trainable vars:')
    for v in trainable_var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    local_init_op = tf.variables_initializer(local_var_list)
    global_init_op = tf.variables_initializer(global_var_list)

    def init_sync_pairs():
        pairs = []
        for v in local_var_list:
            if v.name.startswith("local"):
                global_v_name = v.name.replace('local', 'global', 1)
                for global_v in global_var_list:
                    if global_v.name == global_v_name:
                        pairs.append((v, global_v))
                        break
        return pairs

    init_sync = tf.group(*[v1.assign(v2) for v1, v2 in init_sync_pairs()])

    saver = FastSaver(global_var_list, max_to_keep=3)
    saver_path = os.path.join(logdir, "model.ckpt")
    report_uninitialized_variables = tf.report_uninitialized_variables()

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        logger.info("Uninizialied Variables after init_fn: %s", ses.run(report_uninitialized_variables))

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])

    summary_dir = logdir + "_{}/worker_{}".format(args.task % tasks, int((args.task - (args.task % tasks)) / tasks))
    summary_writer = tf.summary.FileWriter(summary_dir, flush_secs=30)

    logger.info("Events directory: %s", summary_dir)

    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=global_init_op,
                             init_fn=init_fn,
                             local_init_op=local_init_op,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(tf.global_variables()),
                             global_step=trainer.global_step,
                             save_model_secs=120,
                             save_summaries_secs=30,
                             recovery_wait_secs=5
                             )

    num_global_steps = 100000 #20000000

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        uninitialized_variables = sess.run(report_uninitialized_variables)
        if len(uninitialized_variables) > 0:
            logger.info("Some variables are not initialized:\n{}").format(uninitialized_variables)
        assert len(uninitialized_variables) == 0

        sess.run(init_sync)

        if args.task < args.num_workers:
            trainer.start(sess, summary_writer)
            global_step = sess.run(trainer.global_step)
            logger.info("Starting training at step=%d", global_step)
            while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
                trainer.process(sess)
                global_step = sess.run(trainer.global_step)
            if args.task == 0:
                saver.save(sess, saver_path, global_step)

            while not sv.should_stop():
                time.sleep(5)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)
    if args.task == 0:
        with open(os.path.join(original_logdir, 'done.txt'), "w") as file:
            file.write(str(global_step))


def cluster_spec(num_workers, num_ps):
    """
More tensorflow setup for data parallelism
"""
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster


def main(_):
    """
Setting up Tensorflow for data parallel work
"""

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="/tmp/attentive", help='Log directory path')
    parser.add_argument('--env-id', default="grid-worlds-v0,grid-worlds-v1", help='Environment ids')

    args = parser.parse_args()
    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warning('Received signal %s: exiting', signal)
        sys.exit(128 + signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
        run(args, server)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)


if __name__ == "__main__":
    tf.app.run()
