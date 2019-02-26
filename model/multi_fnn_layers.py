from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.layers import base as base_layer

from tensorflow.python.platform import tf_logging as logging

from tensorflow.python.util import nest

from lib import dcrnn_utils

class MultiFNNCell(base_layer.Layer):

    def call(self, inputs, **kwargs):
        pass

    def build(self, _):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, cells):
        super(MultiFNNCell, self).__init__()
        if not cells:
            raise ValueError("Must specify at least one cell for MultiFNNCell")
        if not nest.is_sequence(cells):
            raise TypeError("cells must be a list or tuple, but saw: %s." % cells)
        self._cells = cells

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def __call__(self, inputs, state=None, scope=None):
        cur_inp = inputs
        for i, cell in enumerate(self._cells):
            with tf.variable_scope("cell_%d" % i):
                cur_inp = cell(cur_inp)
                # if i == 0:
                #     cur_inp = tf.layers.dropout(cur_inp, rate=0.2)

        return cur_inp
