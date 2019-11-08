import numpy as np
import tensorflow as tf

"""
This class implements network and also correlation matrix
To prevent gradients from blowing up,
try to use sigmoid as activation rather than relu
"""

class net():
    """
    This is the network architecture
    """
    def __init__(self, model_path=None, trainable=True):
        if model_path is not None:
            self.data_dict = np.load(model_path, encoding='latin1', allow_pickle=True).item()
        else:
            self.data_dict = None
        self.var_dict = {}
        self.trainable = trainable

    def build(self, input):
        """
        1 dim convolution
        :param input:
        :return:
        """
        self.conv = self.conv1d_layer(input, 4, 1, 4, 'conv1d')

        self.output = tf.subtract(self.conv, tf.reduce_mean(self.conv, axis=2, keepdims=True))

        self.cov = tf.matmul(self.output, tf.transpose(self.output, perm=[0, 2, 1]))

        mid = tf.reduce_sum(tf.square(self.output), axis=2, keepdims=True)
        self.std = tf.sqrt(tf.matmul(mid, tf.transpose(mid, perm=[0, 2, 1])))
        self.corr = tf.div(self.cov, self.std)

        self.map = self.corr

        self.data_dict = None

    def conv1d_layer(self, bottom, in_size, in_channels, out_channels, name):
        with tf.variable_scope(name):
            x = tf.reshape(bottom, [-1, in_size, in_channels])
            filt, conv_biases = self.get_conv1d_var(2, in_channels, out_channels, name)
            conv = tf.nn.conv1d(x, filt, [1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            sigmoid = tf.nn.sigmoid(bias)

            return sigmoid

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], .0, .01)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .01)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, .01)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .01)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_conv1d_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, in_channels, out_channels], 0.0, .01)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .01)
        biases = self.get_var(initial_value, name, 1, name + "biases")

        return filters, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name, trainable=True)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./net-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path
