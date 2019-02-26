from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import legacy_seq2seq

from lib.metrics import masked_mse_loss, masked_mae_loss, masked_rmse_loss
from model.dcrnn_cell import DCGRUCell
from model.rnn_cell import GRUCell
from model.tf_model import TFModel


class CLIPSCNNDCRNNModel(TFModel):
    def __init__(self, is_training, config, scaler=None, adj_mx=None):
        super(CLIPSCNNDCRNNModel, self).__init__(config, scaler=scaler)
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
        num_heads = 2
        output_dim = int(config.get('output_dim', 1))
        rnn_units = int(config.get('rnn_units'))
        seq_len = int(config.get('seq_len'))
        clip = int(config.get('clip'))
        use_curriculum_learning = bool(config.get('use_curriculum_learning', False))
        # aux_dim = input_dim - output_dim
        aux_dim = 1 # specially is time_in_day
        # assert input_dim == output_dim, 'input_dim: %d != output_dim: %d' % (input_dim, output_dim)
        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, output_dim + aux_dim), name='labels')

        GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * (output_dim + aux_dim)))

        if max_diffusion_step == 0:
            cell = GRUCell(rnn_units, num_nodes=num_nodes)
            cell_with_projection = GRUCell(rnn_units, num_nodes=num_nodes, num_proj=output_dim)
        else:
            cell = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes, filter_type=filter_type)
            cell_with_projection = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                                         num_proj=output_dim, filter_type=filter_type)
        encoding_cells = [cell] * num_rnn_layers
        decoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]

        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('DCRNN_SEQ'):
            ### 1: CNN 4layers
            #         self._inputs: [batch_size, seq_len, num_nodes, input_dim], input_dim = width_in * in_channels
            #         [batch_size * num_nodes, height_in=seq_len24, width_in, in_channels], Width_in=12, Channels_in=2
            #    CNN1 Filter_width=2, Channels_out=16, Width_out=11
            #    CNN2 Filter_width=4, Channels_out=32, Width_out=8
            #    CNN3 Filter_width=8, Channels_out=64, Width_out=1
            inputs = tf.reshape(tf.transpose(self._inputs, perm=[0, 2, 1, 3]), shape=[batch_size * num_nodes, seq_len, input_dim])
            inputs = tf.expand_dims(inputs, axis=-1)
            dtype = inputs.dtype
            in_width = int(clip)
            in_channels = int(input_dim/clip)
            inputs = tf.reshape(inputs, shape=[batch_size * num_nodes, seq_len, in_channels, in_width])
            inputs = tf.transpose(inputs, perm=[0, 1, 3, 2]) # [batch_size*num_nodes, seq_len, in_width, in_channels]
            # feed in No.1 CNN:
            W1 = tf.get_variable('cnn1_w', [1, 4, in_channels, 128], dtype=dtype,
                                 initializer=tf.contrib.layers.xavier_initializer())
            inputs = tf.nn.conv2d(inputs, W1, strides=[1, 1, 4, 1], padding='VALID')
            b1 = tf.get_variable('cnn1_b', [128], dtype=dtype, initializer=tf.constant_initializer(0.0, dtype=dtype))
            inputs = tf.nn.relu(inputs + b1)
            # feed in No.2 CNN:
            W2 = tf.get_variable('cnn2_w', [1, 2, 128, 128], dtype=dtype,
                                 initializer=tf.contrib.layers.xavier_initializer())
            inputs = tf.nn.conv2d(inputs, W2, strides=[1, 1, 1, 1], padding='VALID')
            b2 = tf.get_variable('cnn2_b', [128], dtype=dtype, initializer=tf.constant_initializer(0.0, dtype=dtype))
            inputs = tf.nn.relu(inputs + b2)   # 128d
            # output_features transform:
            inputs = tf.reshape(inputs, shape=[batch_size, num_nodes, seq_len, -1])
            inputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
            input_dim = inputs.get_shape()[-1].value
            ### 2. connect to DCRNN_seq2seq frame:
            inputs = tf.unstack(tf.reshape(inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)
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
                    result = tf.reshape(result, (batch_size, num_nodes * (output_dim + aux_dim)))
                return result

            # rnn_encoder:
            enc_outputs, enc_states = self.rnn_encoder_variant(inputs, batch_size, encoding_cells)
            # First calculate a concatenation of encoder outputs to put attention on.
            top_states = [tf.reshape(e, [-1, 1, rnn_units]) for e in enc_outputs]
            attention_states = tf.concat(top_states, axis=1)  # [batch_size*num_nodes, seq_len, rnn_units]
            # 2. attention_rnn_decoder:
            attn_length = attention_states.get_shape()[1].value  # seq_len
            attn_size = attention_states.get_shape()[2].value  # rnn_units
            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            hidden = tf.reshape(attention_states, [-1, attn_length, 1, attn_size])
            hidden_features = []
            v = []
            attention_vec_size = attn_size  # Size of query vectors for attention.
            for a in range(num_heads):
                k = tf.get_variable("AttnW_%d" % a, [1, 1, attn_size, attention_vec_size])
                hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
                v.append(tf.get_variable("AttnV_%d" % a, [attention_vec_size]))
            def attention(query):
                """Put attention masks on hidden using hidden_features and query."""
                ds = []  # Results of attention reads will be stored here.
                for a in range(num_heads):
                    with tf.variable_scope("Attention_%d" % a):
                        y = self.linear(query, attention_vec_size)
                        y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                        # Attention mask is a softmax of v^T * tanh(...).
                        y = tf.reduce_sum(v[a] * tf.nn.tanh(hidden_features[a] + y), [2, 3])
                        y = tf.nn.softmax(y)
                        # Now calculate the attention-weighted vector d.
                        y = tf.reduce_sum(tf.reshape(y, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                        ds.append(tf.reshape(y, [-1, attn_size]))
                return ds
            # rnn_decoder:
            dec_outputs, _ = self.attention_rnn_decoder_variant(labels, num_nodes, num_heads, enc_states,
                                                                decoding_cells, loop_function, attention)
            # dec_outputs, _ = self.rnn_decoder_variant(labels, enc_states, decoding_cells, loop_function)
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
