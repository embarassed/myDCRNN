from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.layers import base as base_layer

from tensorflow.python.platform import tf_logging as logging

from lib import dcrnn_utils

class CNNCell(base_layer.Layer):
    """
    自定义多节点的FFNN,多个节点的输入序列同时产生这些节点对应的多个时间戳输出
    """
    def call(self, inputs, **kwargs):
        pass

    def build(self, _):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, filter_size, num_nodes, input_size=None, num_proj=None, activation=tf.nn.relu, reuse=None):
        super(CNNCell, self).__init__(_reuse=reuse)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._filter_size = filter_size

    def __call__(self, inputs, state=None, scope=None, width_stride=None, out_channels=None):
        """
        :param inputs: (B * num_nodes, 1, seq_len, input_dim)
        :return:
        - Ouput: A `2-D`张量,shape=`[batch_size x self.output_size]`
        """
        with tf.variable_scope(scope or "cnn_cell"):
            in_channels = inputs.get_shape()[-1].value
            dtype = inputs.dtype
            weights = tf.get_variable('w', [1, self._filter_size, in_channels, out_channels], dtype=dtype, initializer=tf.contrib.layers.xavier_initializer())
            inputs = tf.nn.conv2d(inputs, weights, strides=[1, 1, width_stride, 1], padding='VALID')
            biases = tf.get_variable("b", [out_channels], dtype=dtype, initializer=tf.constant_initializer(0.0, dtype=dtype))
            output = self._activation(inputs + biases)
        return output


