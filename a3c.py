from __future__ import print_function
from collections import namedtuple
import logging
from model import *
import six.moves.queue as queue
import scipy.signal
import threading

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GAMMA = 0.99

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def process_rollout(rollout, gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, rollout.task)


Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "task"])


class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.task = []

    def add(self, state, action, reward, value, terminal, task):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.task += [task]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.task.extend(other.task)
        self.terminal = other.terminal


class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""

    def __init__(self, env, policy, num_local_steps, task, workers, training_tasks, worker_task):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.worker_task = worker_task
        self.task = task
        self.workers = workers
        self.training_tasks = training_tasks
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.task)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.
            self.queue.put(next(rollout_provider), timeout=600.0)


def env_runner(env, policy, num_local_steps, summary_writer, task):
    """
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread
runner appends the policy to the queue.
"""
    last_state = env.reset()
    length = 0
    rewards = 0

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            fetched = policy.act(last_state, task)
            action, value_ = fetched[0], fetched[1]
            # argmax to convert from one-hot
            state, reward, terminal, info = env.step(action.argmax() % env.action_space.n)

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, task)
            length += 1
            rewards += reward

            last_state = state

            if info and summary_writer is not None:
                summary = tf.Summary()
                for k, v in info.items():
                    if k in ["global/episode_length", "global/episode_reward", "diagnostics/fps",
                             "global/global_step/sec"]:
                        summary.value.add(tag=k, simple_value=float(v))

                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= timestep_limit:
                terminal_end = True
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                logger.info("Episode finished. Sum of rewards: %s. Length: %d" % (rewards, length))
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, task)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout


class A3C(object):
    def __init__(self, env, worker_task, env_task, tasks, ac_spaces, worker_per_task):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""
        self.best_saver = None
        self.env = env
        self.worker_task = worker_task
        self.task = env_task
        self.tasks = tasks
        self.training_tasks = tasks
        self.worker_per_task = worker_per_task
        self.workers = int(self.worker_per_task) * self.training_tasks

        with tf.device(tf.train.replica_device_setter(1, worker_device="/job:ps/task:0")):
            with tf.variable_scope("ps"):
                self.manager_table = tf.contrib.lookup.MutableHashTable(name="manager_table", key_dtype=tf.string,
                                                                        value_dtype=tf.float32, default_value=0.0)

        worker_device = "/job:worker/task:{}/cpu:0".format(worker_task)
        logger.info("Worker device is: {}".format(worker_device))
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = FFPolicy(env.observation_space.shape, ac_spaces, self.tasks)

                self.global_step = tf.get_variable("global_step", [], tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = FFPolicy(env.observation_space.shape, ac_spaces, self.tasks)
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            logits = pi.logits[self.task]
            vf = pi.vf[self.task]

            log_prob_tf = tf.nn.log_softmax(logits)
            prob_tf = tf.nn.softmax(logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            self.loss = pi_loss + 0.5 * vf_loss - 0.02 * entropy

            grads = tf.gradients(self.loss, pi.var_list)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            bs = tf.to_float(tf.shape(pi.x)[0])

            tf.summary.scalar("model/policy_loss", pi_loss / bs)
            tf.summary.scalar("model/value_loss", vf_loss / bs)
            tf.summary.image("model/state", pi.x)
            tf.summary.scalar("model/entropy", entropy / bs)
            tf.summary.scalar("model/logits", tf.reduce_sum(logits) / bs)
            tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
            tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))

            for i in range(pi.nns):
                tf.summary.scalar("model/weight_" + str(i + 1), tf.reduce_sum(tf.gather(tf.transpose(pi.w), [i])) / bs)
            self.summary_op = tf.summary.merge_all()

            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of parameters
            opt = tf.train.AdamOptimizer(learning_rate=0.0001)
            grads_and_vars = list(zip(grads, self.network.var_list))
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)

            self.summary_writer = None
            self.local_steps = 0

            self.runner = RunnerThread(env, pi, 5, self.task, self.workers, self.training_tasks, self.worker_task)

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
self explanatory:  take a rollout from the queue of the thread runner.
"""
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server.
"""
        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=GAMMA, lambda_=1.0)

        should_compute_summary = self.worker_task < self.tasks and self.local_steps >= 11

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
            self.local_steps -= 11
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_network.task: batch.task
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary and self.summary_writer is not None:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()

        self.local_steps += self.worker_per_task
