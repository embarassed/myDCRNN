from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import numpy as np
from tensorflow.contrib.rnn import RNNCell


def modrelu(inputs, bias):
    """
    modReLU activation function
    """

    norm = tf.abs(inputs) + 0.00001
    biased_norm = norm + bias
    magnitude = tf.nn.relu(biased_norm)
    phase = tf.sign(inputs)

    return phase * magnitude


def generate_index_tunable(s, L):
    """
    generate the index lists for goru to prepare orthogonal matrices
    and perform efficient rotations
    This function works for tunable case
    """
    ind1 = list(range(s))
    ind2 = list(range(s))

    for i in range(s):
        if i % 2 == 1:
            ind1[i] = ind1[i] - 1
            if i == s - 1:
                continue
            else:
                ind2[i] = ind2[i] + 1
        else:
            ind1[i] = ind1[i] + 1
            if i == 0:
                continue
            else:
                ind2[i] = ind2[i] - 1

    ind_exe = [ind1, ind2] * int(L / 2)

    ind3 = []
    ind4 = []

    for i in range(int(s / 2)):
        ind3.append(i)
        ind3.append(i + int(s / 2))

    ind4.append(0)
    for i in range(int(s / 2) - 1):
        ind4.append(i + 1)
        ind4.append(i + int(s / 2))
    ind4.append(s - 1)

    ind_param = [ind3, ind4]

    return ind_exe, ind_param


def generate_index_fft(s):
    """
    generate the index lists for goru to prepare orthogonal matrices
    and perform efficient rotations
    This function works for fft case
    """

    def ind_s(k):
        if k == 0:
            return np.array([[1, 0]])
        else:
            temp = np.array(range(2 ** k))
            list0 = [np.append(temp + 2 ** k, temp)]
            list1 = ind_s(k - 1)
            for i in range(k):
                list0.append(np.append(list1[i], list1[i] + 2 ** k))
            return list0

    t = ind_s(int(math.log(s / 2, 2)))

    ind_exe = []
    for i in range(int(math.log(s, 2))):
        ind_exe.append(tf.constant(t[i]))

    ind_param = []
    for i in range(int(math.log(s, 2))):
        ind = np.array([])
        for j in range(2 ** i):
            ind = np.append(ind, np.array(range(0, s, 2 ** i)) + j).astype(np.int32)

        ind_param.append(tf.constant(ind))

    return ind_exe, ind_param


def fft_param(num_units, scope=None):
    with tf.variable_scope(scope):
        phase_init = tf.random_uniform_initializer(-3.14, 3.14)
        capacity = int(math.log(num_units, 2))

        theta = tf.get_variable("theta"+scope, [capacity, num_units // 2],
                                initializer=phase_init)
        cos_theta = tf.cos(theta)
        sin_theta = tf.sin(theta)

        cos_list = tf.concat([cos_theta, cos_theta], axis=1)
        sin_list = tf.concat([sin_theta, -sin_theta], axis=1)

        ind_exe, index_fft = generate_index_fft(num_units)

        v1 = tf.stack([tf.gather(cos_list[i, :], index_fft[i]) for i in range(capacity)])
        v2 = tf.stack([tf.gather(sin_list[i, :], index_fft[i]) for i in range(capacity)])

    return v1, v2, ind_exe


def tunable_param(num_units, capacity, scope=None):
    with tf.variable_scope(scope):
        capacity_A = int(capacity // 2)
        capacity_B = capacity - capacity_A
        phase_init = tf.random_uniform_initializer(-3.14, 3.14)

        theta_A = tf.get_variable("theta_A"+scope, [capacity_A, num_units // 2],
                                  initializer=phase_init)
        cos_theta_A = tf.cos(theta_A)
        sin_theta_A = tf.sin(theta_A)

        cos_list_A = tf.concat([cos_theta_A, cos_theta_A], axis=1)
        sin_list_A = tf.concat([sin_theta_A, -sin_theta_A], axis=1)

        theta_B = tf.get_variable("theta_B"+scope, [capacity_B, num_units // 2 - 1],
                                  initializer=phase_init)
        cos_theta_B = tf.cos(theta_B)
        sin_theta_B = tf.sin(theta_B)

        cos_list_B = tf.concat([tf.ones([capacity_B, 1]), cos_theta_B,
                                cos_theta_B, tf.ones([capacity_B, 1])], axis=1)
        sin_list_B = tf.concat([tf.zeros([capacity_B, 1]), sin_theta_B,
                                - sin_theta_B, tf.zeros([capacity_B, 1])], axis=1)

        ind_exe, [index_A, index_B] = generate_index_tunable(num_units, capacity)

        diag_list_A = tf.gather(cos_list_A, index_A, axis=1)
        off_list_A = tf.gather(sin_list_A, index_A, axis=1)
        diag_list_B = tf.gather(cos_list_B, index_B, axis=1)
        off_list_B = tf.gather(sin_list_B, index_B, axis=1)

        v1 = tf.reshape(tf.concat([diag_list_A, diag_list_B], axis=1), [capacity, num_units])
        v2 = tf.reshape(tf.concat([off_list_A, off_list_B], axis=1), [capacity, num_units])

    return v1, v2, ind_exe


class GORUCell(RNNCell):
    """Gated Orthogonal Recurrent Unit Cell

    The implementation is based on:

    http://arxiv.org/abs/1706.02761.

    """

    def __init__(self, num_units, num_nodes, capacity=8, fft=False, num_proj=None, activation=modrelu, reuse=None, scope=None):
        """Initializes the GORU cell.
        Args:
          num_units: int, The number of units in the GORU cell.
          capacity: int, The capacity of the orthogonal matrix for tunable
            case.
          fft: bool, default false, whether to use fft style
          architecture or tunable style.
        """

        super(GORUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._activation = activation
        self._capacity = capacity
        self._fft = fft

        if self._capacity > self._num_units:
            raise ValueError("Do not set capacity larger than hidden size, it is redundant")

        if self._fft:
            if math.log(self._num_units, 2) % 1 != 0:
                raise ValueError("FFT style only supports power of 2 of hidden size")
        else:
            if self._num_units % 2 != 0:
                raise ValueError("Tunable style only supports even number of hidden size")

            if self._capacity % 2 != 0:
                raise ValueError("Tunable style only supports even number of capacity")

        if self._fft:
            self._capacity = int(math.log(self._num_units, 2))
            self._v1, self._v2, self._ind = fft_param(self._num_units, scope=scope)
        else:
            self._v1, self._v2, self._ind = tunable_param(self._num_units, self._capacity, scope=scope)

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def loop(self, h):
        for i in range(self._capacity):
            diag = h * self._v1[i, :]
            off = h * self._v2[i, :]
            h = diag + tf.gather(off, self._ind[i], axis=1)

        return h

    def __call__(self, inputs, state, scope=None):
        # inputs: [batch_size, num_nodes * input_dim]
        # state: [batch_size, state_size] state_size=num_nodes*input_dim
        with tf.variable_scope(scope or "goru_cell"):
            # input_matrix_init = tf.random_uniform_initializer(-0.01, 0.01)
            batch_size = inputs.get_shape()[0].value
            bias_init = tf.constant_initializer(2.)
            mod_bias_init = tf.constant_initializer(0.01)

            # U = tf.get_variable("U", [inputs_size, self._num_units * 3], dtype=tf.float32,
            #                     initializer=input_matrix_init)
            # Ux = tf.matmul(inputs, U)
            with tf.variable_scope("ux"):
                Ux = self._gconv(inputs, self._num_units * 3)
            U_cx, U_rx, U_gx = tf.split(Ux, num_or_size_splits=3, axis=1)  #[batch_size * num_nodes, state_dim]
            with tf.variable_scope("wh"):
                Ws = self._gconv(state, self._num_units * 2)
            W_rh, W_gh = tf.split(Ws, num_or_size_splits=2, axis=1)   #[batch_size * num_nodes, state_dim]

            bias_r = tf.get_variable("bias_r", [self._num_units], dtype=tf.float32, initializer=bias_init)
            bias_g = tf.get_variable("bias_g", [self._num_units], dtype=tf.float32)
            bias_c = tf.get_variable("bias_c", [self._num_units], dtype=tf.float32, initializer=mod_bias_init)

            r_tmp = U_rx + W_rh + bias_r
            g_tmp = U_gx + W_gh + bias_g
            r = tf.nn.sigmoid(r_tmp)
            g = tf.nn.sigmoid(g_tmp)

            Unitaryh = self.loop(tf.reshape(state, shape=[batch_size * self._num_nodes, -1]))
            c = self._activation(r * Unitaryh + U_cx, bias_c)
            c = tf.reshape(c, shape=[batch_size, -1])
            g = tf.reshape(g, shape=[batch_size, -1])
            new_state = tf.multiply(g, state) + tf.multiply(1 - g, c)
            output = new_state  #[batch_size, num_nodes*num_units]

            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self._num_proj), dtype=inputs.dtype, initializer=tf.contrib.layers.xavier_initializer())
                    output = tf.reshape(output, shape=(-1, self._num_units))
                    output = tf.reshape(tf.matmul(output, w), shape=[batch_size, self.output_size])

        return output, new_state  #[batch_size, num_nodes*num_units]

    def _gconv(self, inputs, output_size, scope=None):
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
        input_size = inputs.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size])

            weights = tf.get_variable(
                'weights', [input_size, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            # biases = tf.get_variable(
            #     "biases", [output_size],
            #     dtype=dtype,
            #     initializer=tf.constant_initializer(bias_start, dtype=dtype))
            # x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size * num_node, state_dim) -> (batch_size, num_node * state_dim)
        # return tf.reshape(x, [batch_size, self._num_nodes * output_size])
        return x    #[batch_size * num_nodes, state_dim]






