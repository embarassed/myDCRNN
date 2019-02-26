from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.contrib import legacy_seq2seq

from lib.metrics import masked_mse_loss, masked_mae_loss, masked_rmse_loss
from model.psrnn_cell import PSRNNCell
from model.tf_model import TFModel


class PSRNNModel(TFModel):
    def __init__(self, is_training, config, params, scaler=None, adj_mx=None):
        super(PSRNNModel, self).__init__(config, scaler=scaler)
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
        output_dim = int(config.get('output_dim', 1))
        rnn_units = int(config.get('rnn_units'))
        seq_len = int(config.get('seq_len'))
        use_curriculum_learning = bool(config.get('use_curriculum_learning', False))
        # aux_dim = input_dim - output_dim
        aux_dim = 1 # specially is time_in_day
        # assert input_dim == output_dim, 'input_dim: %d != output_dim: %d' % (input_dim, output_dim)
        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, output_dim + aux_dim), name='labels')

        # random fourier features
        W_rff = tf.get_variable('W_rff', initializer=tf.constant(params.W_rff.astype(np.float32)), dtype=tf.float32)
        b_rff = tf.get_variable('b_rff', initializer=tf.constant(params.b_rff.astype(np.float32)), dtype=tf.float32)

        self._W_rff = W_rff
        self._b_rff = b_rff

        z = tf.tensordot(tf.reshape(self._inputs, [batch_size, seq_len, input_dim * num_nodes]), W_rff, axes=[[2], [0]]) + b_rff
        inputs_rff = tf.cos(z) * np.sqrt(2.) / np.sqrt(config.get('nRFF_Obs'))

        # dimensionality reduction
        U = tf.get_variable('U', initializer=tf.constant(params.U.astype(np.float32)), dtype=tf.float32)
        U_bias = tf.get_variable('U_bias', [config.hidden_size], initializer=tf.constant_initializer(0.0))
        inputs_embed = tf.tensordot(inputs_rff, U, axes=[[2], [0]]) + U_bias

        GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * (output_dim + aux_dim)))
        z = tf.tensordot(GO_SYMBOL, W_rff, axes=[[1], [0]]) + b_rff
        GO_SYMBOL = tf.cos(z) * np.sqrt(2.) / np.sqrt(config.get('nRFF_Obs'))
        GO_SYMBOL = tf.tensordot(GO_SYMBOL, U, axes=[[1], [0]]) + U_bias

        z = tf.tensordot(tf.reshape(self._labels, [batch_size, seq_len, (output_dim + aux_dim) * num_nodes]), W_rff,
                         axes=[[2], [0]]) + b_rff
        inputs_rff = tf.cos(z) * np.sqrt(2.) / np.sqrt(config.get('nRFF_Obs'))

        # dimensionality reduction
        U = tf.get_variable('U', initializer=tf.constant(params.U.astype(np.float32)), dtype=tf.float32)
        U_bias = tf.get_variable('U_bias', [config.hidden_size], initializer=tf.constant_initializer(0.0))
        inputs_embed = tf.tensordot(inputs_rff, U, axes=[[2], [0]]) + U_bias

        # outputs, state = tf.contrib.rnn.static_rnn(
        #     cell, inputs_unstacked, initial_state=self._initial_state)
        #
        # # reshape output
        # output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
        # ###################################################################################################
        #
        #
        cell = PSRNNCell(rnn_units, params=params)
        cell_with_projection = PSRNNCell(rnn_units, params=params, num_proj=output_dim)
        encoding_cells = [cell] * num_rnn_layers
        decoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]

        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes * output_dim)
        with tf.variable_scope('DCRNN_SEQ'):
            inputs = tf.unstack(inputs_embed, axis=1)
            labels = tf.unstack(
                tf.reshape(self._labels[..., :output_dim], (batch_size, horizon, num_nodes * output_dim)), axis=1)
            if aux_dim > 0:
                aux_info = tf.unstack(self._labels[..., output_dim:], axis=1)
                aux_info.insert(0, None)
            # labels.insert(0, GO_SYMBOL)

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
                    result = tf.reshape(result, (batch_size, num_nodes * (output_dim + aux_dim)))
                    z = tf.tensordot(result, W_rff, axes=[[1], [0]]) + b_rff
                    result_rff = tf.cos(z) * np.sqrt(2.) / np.sqrt(config.get('nRFF_Obs'))
                    # dimensionality reduction
                    result = tf.tensordot(result_rff, U, axes=[[1], [0]]) + U_bias
                return result

            # rnn_encoder:
            _, enc_states = self.rnn_encoder(inputs, batch_size, encoding_cells)
            # rnn_decoder:
            dec_outputs, _ = self.rnn_decoder_psrnn(GO_SYMBOL, labels, enc_states, decoding_cells, loop_function)
        outputs = dec_outputs

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
