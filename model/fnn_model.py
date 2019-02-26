from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lib.metrics import masked_mse_loss, masked_mae_loss, masked_rmse_loss
from model.fnn_cell import FNNCell
from model.multi_fnn_layers import MultiFNNCell
from model.tf_model import TFModel


class FNNModel(TFModel):
    def __init__(self, is_training, config, scaler=None, adj_mx=None):
        super(FNNModel, self).__init__(config, scaler=scaler)
        batch_size = int(config.get('batch_size'))
        horizon = int(config.get('horizon', 1))
        cl_decay_steps = int(config.get('cl_decay_steps', 1000))
        input_dim = int(config.get('input_dim', 1))
        loss_func = config.get('loss_func', 'MSE')
        max_grad_norm = float(config.get('max_grad_norm', 5.0))
        num_nodes = int(config.get('num_nodes', 1))
        num_fnn_layers = int(config.get('num_fnn_layers', 1))
        output_dim = int(config.get('output_dim', 1))
        fnn_units = int(config.get('fnn_units'))
        seq_len = int(config.get('seq_len'))


        use_curriculum_learning = bool(config.get('use_curriculum_learning', False))
        # aux_dim = input_dim - output_dim
        aux_dim = 1  # specially is time_in_day
        # assert input_dim == output_dim, 'input_dim: %d != output_dim: %d' % (input_dim, output_dim)
        # Input (batch_size, timesteps, num_nodes, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Labels: (batch_size, timesteps, num_nodes, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, output_dim + aux_dim), name='labels')

        GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes, (output_dim + aux_dim)))

        cell = FNNCell(fnn_units, num_nodes=num_nodes)
        cell_with_projection = FNNCell(fnn_units, num_nodes=num_nodes, num_proj=output_dim)

        layers = [cell] * (num_fnn_layers - 1) + [cell_with_projection]

        fnn_layers = MultiFNNCell(layers)

        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('FNN_SEQ'):
            inputs = tf.unstack(self._inputs, axis=1)
            labels = tf.unstack(self._labels[..., :output_dim], axis=1)
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
                    result = tf.concat([result, aux_info[i]], axis=-1)
                return result

            outputs = self.mlp_endecoder(labels, inputs, fnn_layers, loop_function)
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
