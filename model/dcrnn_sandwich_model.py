from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import legacy_seq2seq

from lib.metrics import masked_mse_loss, masked_mae_loss, masked_rmse_loss
from model.dcrnn_cell import DCGRUCell
from model.gcn_cell import GCNCell
from model.multi_gcn_layers import MultiGCNCell
from model.tf_model import TFModel
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.rnn import _rnn_step
from tensorflow.python.ops.rnn import _infer_state_dtype

# pylint: disable=protected-access
_concat = rnn_cell_impl._concat
_like_rnncell = rnn_cell_impl._like_rnncell


class DCRNNSandwichModel(TFModel):
    def __init__(self, is_training, config, scaler=None, adj_mx=None):
        super(DCRNNSandwichModel, self).__init__(config, scaler=scaler)
        batch_size = int(config.get('batch_size'))
        max_diffusion_step = int(config.get('max_diffusion_step', 2))
        cl_decay_steps = int(config.get('cl_decay_steps', 1000))
        filter_type = config.get('filter_type', 'laplacian')
        horizon = int(config.get('horizon', 1))
        input_dim = int(config.get('input_dim', 1))
        loss_func = config.get('loss_func', 'MSE')
        max_grad_norm = float(config.get('max_grad_norm', 5.0))
        num_nodes = int(config.get('num_nodes', 1))
        num_rnn_layers = int(config.get('num_rnn_layers', 1))
        num_gcn_layers = int(config.get('num_gcn_layers', 1))
        output_dim = int(config.get('output_dim', 1))
        rnn_units = int(config.get('rnn_units'))
        gcn_units = int(config.get('gcn_units'))
        seq_len = int(config.get('seq_len'))
        use_curriculum_learning = bool(config.get('use_curriculum_learning', False))
        aux_dim = input_dim - output_dim
        # assert input_dim == output_dim, 'input_dim: %d != output_dim: %d' % (input_dim, output_dim)
        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, input_dim), name='labels')

        GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * input_dim))

        gcncell = GCNCell(gcn_units, adj_mx=adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                          filter_type=filter_type)
        gcn_layers_ = [gcncell] * num_gcn_layers
        gcn_layers = MultiGCNCell(gcn_layers_)

        encoding_cells = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                                   filter_type=filter_type)
        decoding_cells = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                                   filter_type=filter_type)
        decoding_cells_with_projection = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step,
                                                   num_nodes=num_nodes,
                                                   num_proj=output_dim, filter_type=filter_type)
        # encoding_cells = [cell] * num_rnn_layers
        # decoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]
        # encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)
        # decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=True)

        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('DCRNN_SEQ'):
            inputs = tf.unstack(tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)
            labels = tf.unstack(
                tf.reshape(self._labels[..., :output_dim], (batch_size, horizon, num_nodes * output_dim)), axis=1)
            if aux_dim > 0:
                aux_info = tf.unstack(self._labels[..., output_dim:], axis=1)
                aux_info.insert(0, None)
            labels.insert(0, GO_SYMBOL)

            def loop_function(prev, i):
                if is_training:
                    # Return either the model's prediction or the previous ground truth in training.
                    if use_curriculum_learning:
                        c = tf.random_uniform((), minval=0, maxval=1.)
                        threshold = self._compute_sampling_threshold(global_step, cl_decay_steps)
                        result = tf.cond(tf.less(c, threshold), lambda: labels[i], lambda: prev)
                    else:
                        result = labels[i]
                else:
                    # Return the prediction of the model in testing.
                    result = prev
                if aux_dim > 0:
                    result = tf.reshape(result, (batch_size, num_nodes, output_dim))
                    result = tf.concat([result, aux_info[i]], axis=-1)
                    result = tf.reshape(result, (batch_size, num_nodes * input_dim))
                return result

            # _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            # enc_outputs, _ = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            # 1 rnn_encoder:
            for cell_layer in range(num_rnn_layers):
                with tf.variable_scope('rnn_encoder_%d' % cell_layer):
                    if cell_layer == 0:
                        enc_outputs, state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
                    else:
                        enc_outputs, state = tf.contrib.rnn.static_rnn(encoding_cells, enc_outputs, initial_state=state,
                                                                       dtype=tf.float32)
            # 2 gcn_layer:
            gcn_outputs = gcn_layers(enc_outputs[-1])
            # 3 rnn_decoder:
            # decoder_state = state
            with tf.variable_scope("rnn_decoder"):
                decoder_state = gcn_outputs
                outputs = []
                prev = None
                for i, inp in enumerate(labels):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    if loop_function is not None and prev is not None:
                        with tf.variable_scope("loop_function", reuse=True):
                            inp = loop_function(prev, i)
                    # output, state = cell(inp, state)
                    for cell_layer in range(num_rnn_layers - 1):
                        with tf.variable_scope('cell_%d' % cell_layer) as cellvs:
                            if i > 0:
                                cellvs.reuse_variables()
                            if cell_layer == 0:
                                dec_outputs, state = decoding_cells(inp, decoder_state)
                            else:
                                dec_outputs, state = decoding_cells(dec_outputs, state)
                    with tf.variable_scope('cell_%d' % (num_rnn_layers - 1)) as cell_vs:
                        if i > 0:
                            cell_vs.reuse_variables()
                        dec_outputs, state = decoding_cells_with_projection(dec_outputs, state)
                    outputs.append(dec_outputs)
                    if loop_function is not None:
                        prev = dec_outputs
            # with tf.variable_scope('rnn_decoder'):
            #     outputs, final_state = legacy_seq2seq.rnn_decoder(labels, enc_outputs[-1], decoding_cells, loop_function=loop_function)

        # Project the output to output_dim.
        outputs = tf.stack(outputs[:-1], axis=1)
        self._outputs = tf.reshape(outputs, (batch_size, horizon, num_nodes, output_dim), name='outputs')

        # preds = self._outputs[..., 0]
        preds = self._outputs
        labels = self._labels[..., :output_dim]

        null_val = config.get('null_val', 0.)
        self._mae = masked_mae_loss(self._scaler, null_val)(preds=preds, labels=labels)

        if loss_func == 'MSE':
            self._loss = masked_mse_loss(self._scaler, null_val)(preds=preds, labels=labels)
        elif loss_func == 'MAE':
            self._loss = masked_mae_loss(self._scaler, null_val)(preds=preds, labels=labels)
        elif loss_func == 'RMSE':
            self._loss = masked_rmse_loss(self._scaler, null_val)(preds=preds, labels=labels)
        else:
            self._loss = masked_mse_loss(self._scaler, null_val)(preds=preds, labels=labels)
        if is_training:
            optimizer = tf.train.AdamOptimizer(self._lr)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self._loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')

        self._merged = tf.summary.merge_all()

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)
