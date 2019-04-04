import numpy as np
import tensorflow as tf

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None,
           trainable=True):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections, trainable=trainable)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections, trainable=trainable)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def linear(x, size, name, initializer=None, bias_init=0, trainable=True):
    w = tf.get_variable(name + "/W", [x.get_shape()[1], size], initializer=initializer, trainable=trainable)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init), trainable=trainable)
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)

def policy_distribution(logits):
    return tf.nn.softmax(logits)


class FFPolicy(object):
    def __init__(self, ob_space, ac_spaces, tasks):
        self.x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.task = tf.placeholder(tf.uint8, [None])
        self.all_entropy = []
        self.all_logits = []
        self.all_actions = []
        self.all_vfs = []

        self.nns = int((tasks + 2) / 4) + 1

        x = tf.nn.relu(conv2d(self.x, 32, "c1", [3, 3], [2, 2]))
        x = tf.nn.relu(conv2d(x, 32, "c2", [3, 3], [1, 1]))
        shared_layer = x

        for i in range(self.nns + 1):
            with tf.variable_scope("nn_" + str(i)):

                x = tf.nn.relu(conv2d(shared_layer, 16, "c3", [3, 3], [1, 1]))
                x = flatten(x)

                if i == self.nns:
                    x = tf.nn.relu(linear(x, self.nns * tasks, "task_in", normalized_columns_initializer(0.01)))
                    one_hot_task = tf.one_hot(self.task, tasks)
                    x = tf.concat([x, one_hot_task], -1)

                x = tf.nn.relu(linear(x, 256, "h1", normalized_columns_initializer(0.01)))

                if i < self.nns:
                    self.all_logits.append(linear(x, max(ac_spaces), "action", normalized_columns_initializer(0.01)))
                    self.all_actions.append(tf.nn.softmax(self.all_logits[-1]))
                    self.all_vfs.append(tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1]))
                else:
                    self.w_logits = linear(x, self.nns, "attention", normalized_columns_initializer(0.01))
                    self.w = tf.nn.softmax(self.w_logits)
                    self.logits = []
                    self.vf = []
                    self.sample = []
                    self.evaluation_policy_dist = []
                    logits = tf.log(
                        tf.clip_by_value(tf.einsum('ij,jik->ik', self.w, tf.convert_to_tensor(self.all_actions)), 1e-8,
                                         1e+8))
                    vf = tf.einsum('ij,ji->i', self.w, tf.convert_to_tensor(self.all_vfs))
                    for j in range(tasks):
                        with tf.variable_scope("task_" + str(j)):
                            x = linear(logits, ac_spaces[j], "logits", normalized_columns_initializer(0.01))
                            self.logits.append(x)
                            self.vf.append(vf)
                            self.sample.append(categorical_sample(self.logits[-1], ac_spaces[j])[0, :])
                            self.evaluation_policy_dist.append(policy_distribution(self.logits[-1]))

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def act(self, ob, task):
        sess = tf.get_default_session()
        result = sess.run([self.sample[task], self.vf[task]], {self.x: [ob], self.task: [task]})
        result[1] = result[1][0]
        return result

    def value(self, ob, task):
        sess = tf.get_default_session()
        return sess.run(self.vf[task], {self.x: [ob], self.task: [task]})[0]

    def evaluation(self, ob, task):
        sess = tf.get_default_session()
        return sess.run([self.sample[task], self.vf[task], self.logits[task],
                         self.evaluation_policy_dist[task]], {self.x: [ob], self.task: [task]})