from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell

from tensorflow.python.platform import tf_logging as logging

from lib import dcrnn_utils


class PSRNNCell(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell.
    """

    def __init__(self, num_units, params, input_size=None, num_proj=None,
                 activation=tf.nn.tanh, reuse=None, filter_type="laplacian"):
        super(PSRNNCell, self).__init__(_reuse=reuse)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._params = params
        self._activation = activation
        self._num_nodes = 1
        self._num_proj = num_proj
        self._max_diffusion_step = 0
        self._supports = []
        supports = []
        # if filter_type == "laplacian":
        #     supports.append(dcrnn_utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        # elif filter_type == "random_walk":
        #     supports.append(dcrnn_utils.calculate_random_walk_matrix(adj_mx).T)
        # elif filter_type == "dual_random_walk":
        #     supports.append(dcrnn_utils.calculate_random_walk_matrix(adj_mx).T)
        #     supports.append(dcrnn_utils.calculate_random_walk_matrix(adj_mx.T).T)
        # else:
        #     supports.append(dcrnn_utils.calculate_scaled_laplacian(adj_mx))
        # for support in supports:
        #     self._supports.append(self._build_sparse_matrix(support))

    def call(self, inputs, **kwargs):
        pass

    def _compute_output_shape(self, input_shape):
        pass

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def __call__(self, inputs, state, scope=None):
        """
         :param inputs: (B, num_nodes * input_dim)

         :return
          - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
          - New state: Either a single `2-D` tensor, or a tuple of tensors matching
          the arity and shapes of `state`
        """
        dtype = inputs.dtype
        batch_size = inputs.shape[0]
        input_dim = inputs.shape[-1]
        inputs = tf.reshape(inputs, [batch_size * self._num_nodes, input_dim])
        state = tf.reshape(state, [batch_size * self._num_nodes, -1])
        with tf.variable_scope(scope or "psrnn_cell"):
            weights_initializer = tf.constant(self._params.W_FE_F.T.astype(np.float32))
            weights = tf.get_variable("weights", initializer=weights_initializer)
            bias_initializer = tf.constant(self._params.b_FE_F.T.astype(np.float32))
            biases = tf.get_variable("bias", initializer=bias_initializer)
            W = tf.add(tf.matmul(state, weights), biases)
            batchedW = tf.split(W, W.shape[0], 0)
            batchedInputs = tf.split(inputs, inputs.shape[0], 0)
            bached_output = []
            for W, input in zip(batchedW, batchedInputs):
                W_square = tf.transpose(tf.reshape(W, [self._num_units, self._num_units]))
                new_s = tf.matmul(input, W_square)
                new_s_normalized = new_s / tf.norm(new_s)
                bached_output.append(new_s_normalized)
            output = tf.concat(bached_output, 0)
            new_state = output = tf.reshape(output, [batch_size, -1])
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', initializer=tf.constant(self._params.W_pred.T.astype(np.float32)))
                    b = tf.get_variable("b", initializer=tf.constant(self._params.b_pred.T.astype(np.float32)), dtype=dtype)

                    output = tf.reshape(output, shape=(-1, self._num_units))
                    output = tf.reshape(tf.nn.bias_add(tf.matmul(output, w), b), shape=(batch_size, self.output_size))
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)
