from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell

from tensorflow.python.platform import tf_logging as logging


class LSTMCell(RNNCell):
    """Graph Convolution Long short-term memory cell.
    """

    def call(self, inputs, **kwargs):
        pass

    def _compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, num_nodes, forget_bias=1.0, input_size=None, num_proj=None,
                 activation=tf.nn.tanh, reuse=None):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param input_size:
        :param num_proj:
        :param activation:
        :param reuse:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        """
        super(LSTMCell, self).__init__(_reuse=reuse)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._forget_bias = forget_bias
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @property
    def state_size(self):
        state_size = 2 * self._num_nodes * self._num_units
        # if self._num_proj is not None:
        #     state_size = self._num_nodes * (self._num_units + self._num_proj)
        return state_size

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory (LSTM) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)

        :return
        - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
        - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`
        """
        with tf.variable_scope(scope or "lstm_cell"):
            with tf.variable_scope("gates"):
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)
                concat = self._gconv(inputs, h, 4 * self._num_units, bias_start=0.0, scope=scope)
                # i = input_gate, j = new_input, f = forget_gate, o = output_gate
                i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)
                # new_c = (c * tf.nn.sigmoid(f) + tf.nn.sigmoid(i) * self._activation(j))
                new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)
            new_state = tf.concat([new_c, new_h], 1)
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self._num_proj), dtype=inputs.dtype,
                                        initializer=tf.contrib.layers.xavier_initializer())
                    batch_size = inputs.get_shape()[0].value
                    new_h = tf.reshape(new_h, shape=(-1, self._num_units))
                    new_h = tf.reshape(tf.matmul(new_h, w), shape=(batch_size, self.output_size))
                    # new_state = tf.concat([new_c, new_h], 1)
            output = new_h
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _gconv(self, inputs, state, output_size, bias_start=0.0, scope=None):
        """Graph convolution between input and the graph matrix.

        :param args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = tf.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs_and_state

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size])

            weights = tf.get_variable(
                'weights', [input_size, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable(
                "biases", [output_size],
                dtype=dtype,
                initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [batch_size, self._num_nodes * output_size])
