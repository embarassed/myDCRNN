from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.layers import base as base_layer

from tensorflow.python.platform import tf_logging as logging

from lib import dcrnn_utils

class FNNFeatureCell(base_layer.Layer):
    """
    自定义多节点的FFNN,多个节点的输入序列同时产生这些节点对应的多个时间戳输出
    """
    def call(self, inputs, **kwargs):
        pass

    def build(self, _):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, num_nodes, input_size=None, num_proj=None, activation=tf.nn.relu, reuse=None):
        super(FNNFeatureCell, self).__init__(_reuse=reuse)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units

    @property
    def output_size(self):
        output_size = self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def __call__(self, inputs, state=None, scope=None):
        """

        :param inputs: (B, num_nodes * input_dim)
        :param state: always None (not use)
        :param scope:
        :return:
        - Ouput: A `2-D`张量,shape=`[batch_size x self.output_size]`
        """
        batch_size = inputs.get_shape()[0].value
        # inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        # input_size = inputs.get_shape()[2].value
        input_size = inputs.get_shape()[1].value
        dtype = inputs.dtype
        x = inputs
        # reshape input to (batch_size, num_nodes, input_dim)
        # x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size])

        with tf.variable_scope(scope or "fnn_cell"):
            with tf.variable_scope("relu_dense"):
                weights = tf.get_variable(
                    'weights', [input_size, self._num_units], dtype=dtype,
                    initializer=tf.constant_initializer(1.0, dtype=dtype))
                x = tf.matmul(x, weights)
                # reshape x back to 2D: (batch_size, num_node, num_units) -> (batch_size, num_node*num_units)
                # x = tf.reshape(x, [batch_size, self._num_nodes * self._num_units])
            # output = self._activation(x)
            output = x
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self.output_size), dtype=dtype,
                                        initializer=tf.constant_initializer(1.0, dtype=dtype))
                    output = tf.matmul(output, w)
                    # output = tf.reshape(tf.matmul(output, w), shape=(batch_size, self.output_size))
        return output


