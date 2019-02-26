from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lib.metrics import masked_mse_loss, masked_mae_loss, masked_rmse_loss
from model.fnn_spacefeature_cell import FNNFeatureCell
from model.tf_model import TFModel


class FNNFeatureModel(TFModel):
    def __init__(self, is_training, config, scaler=None, adj_mx=None):
        super(FNNFeatureModel, self).__init__(config, scaler=scaler)
        batch_size = int(config.get('batch_size'))
        horizon = int(config.get('horizon', 1))
        input_dim = int(config.get('input_dim', 1))
        loss_func = config.get('loss_func', 'MSE')
        max_grad_norm = float(config.get('max_grad_norm', 5.0))
        num_nodes = int(config.get('num_nodes', 1))
        num_fnn_layers = int(config.get('num_fnn_layers', 1))
        output_dim = int(config.get('output_dim', 1))
        fnn_units = int(config.get('fnn_units'))
        seq_len = int(config.get('seq_len'))
        aux_dim = input_dim - output_dim
        # assert input_dim == output_dim, 'input_dim: %d != output_dim: %d' % (input_dim, output_dim)
        # Input (batch_size, timesteps, num_nodes, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Labels: (batch_size, timesteps, num_nodes, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, input_dim), name='labels')
        # reshape input and labels to (batch_size, num_nodes, timesteps * input_dim)
        #  and update input_dim
        inputs = tf.transpose(self._inputs, perm=[0,2,1,3])
        inputs = tf.reshape(inputs, shape=[batch_size, num_nodes, -1])
        self._input_features = inputs.get_shape()[2].value
        self._output_features = horizon * output_dim

        cell_with_projection = FNNFeatureCell(fnn_units, num_nodes=num_nodes, num_proj=self._output_features)

        fnn_layers = cell_with_projection

        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('FNN'):
            inputs = tf.reshape(inputs, (batch_size, num_nodes*self._input_features))
            outputs = fnn_layers(inputs)
        # Project the output to output_dim.
        outputs = tf.reshape(outputs, (batch_size, num_nodes, self._output_features))
        outputs = tf.reshape(outputs, (batch_size, num_nodes, horizon, output_dim))
        self._outputs = tf.transpose(outputs, perm=[0, 2, 1, 3], name='outputs') # (batch_size, horizon, num_nodes, output_dim)

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
