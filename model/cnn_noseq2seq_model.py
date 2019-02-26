from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import legacy_seq2seq

from lib.metrics import masked_mse_loss, masked_mae_loss, masked_rmse_loss
from model.cnn_cell import CNNCell
from model.tf_model import TFModel


class CNNModel(TFModel):
    def __init__(self, is_training, config, scaler=None, adj_mx=None):
        super(CNNModel, self).__init__(config, scaler=scaler)
        batch_size = int(config.get('batch_size'))
        max_diffusion_step = int(config.get('max_diffusion_step', 2))
        cl_decay_steps = int(config.get('cl_decay_steps', 1000))
        filter_type = config.get('filter_type', 'laplacian')
        horizon = int(config.get('horizon', 1))
        input_dim = int(config.get('input_dim', 1))
        loss_func = config.get('loss_func', 'MSE')
        max_grad_norm = float(config.get('max_grad_norm', 5.0))
        num_nodes = int(config.get('num_nodes', 1))
        num_cnn_layers = int(config.get('num_cnn_layers', 1))
        filter_size = int(config.get('filter_size'))
        out_channels = int(config.get('out_channels'))
        output_dim = int(config.get('output_dim', 1))
        # rnn_units = int(config.get('rnn_units'))
        seq_len = int(config.get('seq_len'))
        use_curriculum_learning = bool(config.get('use_curriculum_learning', False))
        # aux_dim = input_dim - output_dim
        aux_dim = 1 # specially is time_in_day
        # assert input_dim == output_dim, 'input_dim: %d != output_dim: %d' % (input_dim, output_dim)
        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, output_dim + aux_dim), name='labels')

        dtype = self._inputs.dtype

        # GO_SYMBOL = tf.zeros(shape=(batch_size * num_nodes, 1, (output_dim + aux_dim)))

        cell = CNNCell(filter_size=filter_size, num_nodes=num_nodes)
        # cell_with_projection = CNNCell(filter_size=filter_size, num_nodes=num_nodes, num_proj=output_dim)
        cells = [cell] * num_cnn_layers

        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, seq_len, num_nodes, output_dim)
        with tf.variable_scope('CNN_SEQ'):
            inputs = tf.transpose(self._inputs, perm=[0, 2, 1, 3]) #[batch_size, num_nodes=height, seq_len=weight, in_channels]
            # CNN layers:
            with tf.variable_scope('CNN') as cnn_vs:
                width_strides = []
                out_channels_list = []
                new_len = inputs.get_shape()[2].value
                for i in range(num_cnn_layers):
                    out_channels_list.append(out_channels)
                    if new_len % 2 != 0 or i == num_cnn_layers - 1:
                        width_strides.append(1)
                    else:
                        width_strides.append(2)
                    new_len = (new_len - filter_size) / width_strides[-1] + 1
                out_channels_list[0] = 32
                out_channels_list[-1] = 128
                for i in range(num_cnn_layers):
                    with tf.variable_scope("cnn_%d" % i):
                        if i == 0:
                            output = cells[i](inputs, out_channels=out_channels_list[i], width_stride=width_strides[i])
                        else:
                            output = cells[i](output, out_channels=out_channels_list[i], width_stride=width_strides[i])
                # output: [batch_size, num_nodes, features, out_channels_list[-1]]
                features = output.get_shape()[2].value
                output = tf.reshape(output, shape=[batch_size * num_nodes, features * out_channels_list[-1]])  # [batch_size*num_nodes, -1]
                out_features = output.get_shape()[-1].value
                # add a mlp:
                w_mlp = tf.get_variable('w', shape=(out_features, 256), dtype=dtype, initializer=tf.contrib.layers.xavier_initializer())
                b_mlp = tf.get_variable('b', [256], dtype=dtype, initializer=tf.constant_initializer(0.0, dtype=dtype))
                output = tf.nn.relu(tf.matmul(output, w_mlp) + b_mlp) #[batch_size*num_nodes, 256]
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(256, horizon * output_dim), dtype=dtype, initializer=tf.contrib.layers.xavier_initializer())
                    output = tf.matmul(output, w)
        # Project the output to output_dim.
        outputs = tf.reshape(output, (batch_size, num_nodes, horizon, output_dim))
        self._outputs = tf.transpose(outputs, perm=[0, 2, 1, 3], name='outputs')  # (batch_size, horizon, num_nodes, output_dim)

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
