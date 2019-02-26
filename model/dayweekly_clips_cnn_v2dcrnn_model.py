from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import legacy_seq2seq

from lib.metrics import masked_mse_loss, masked_mae_loss, masked_rmse_loss
from model.dcrnn_cell import DCGRUCell
from model.rnn_cell import GRUCell
from model.dayweekly_2tf_model import TFModel


class CLIPSCNNDCRNNModel(TFModel):
    def __init__(self, is_training, config, scaler=None, adj_mx=None, cnn_activation=tf.nn.relu):
        super(CLIPSCNNDCRNNModel, self).__init__(config, scaler=scaler)
        batch_size = int(config.get('batch_size'))
        max_diffusion_step = int(config.get('max_diffusion_step', 2))
        cl_decay_steps = int(config.get('cl_decay_steps', 1000))
        filter_type = config.get('filter_type', 'laplacian')
        horizon = int(config.get('horizon', 1))
        input_dim = int(config.get('input_dim', 1))
        dayly_len = int(config.get('dayly_len'))
        weekly_len = int(config.get('weekly_len'))
        loss_func = config.get('loss_func', 'MSE')
        max_grad_norm = float(config.get('max_grad_norm', 5.0))
        num_nodes = int(config.get('num_nodes', 1))
        num_rnn_layers = int(config.get('num_rnn_layers', 1))
        output_dim = int(config.get('output_dim', 1))
        rnn_units = int(config.get('rnn_units'))
        seq_len = int(config.get('seq_len'))
        # small_seq_len = int(config.get('small_seq_len'))
        # if small_seq_len != seq_len:
        #     small_seq_len = seq_len
        clip = int(config.get('clip'))
        use_curriculum_learning = bool(config.get('use_curriculum_learning', False))
        # aux_dim = input_dim - output_dim
        aux_dim = 1 # specially is time_in_day
        # assert input_dim == output_dim, 'input_dim: %d != output_dim: %d' % (input_dim, output_dim)
        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs_small = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, 2), name='inputs_small')
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        self._inputs_dayly = tf.placeholder(tf.float32, shape=(batch_size, horizon, dayly_len, num_nodes, int(input_dim/clip)), name='inputs_dayly')
        self._inputs_weekly = tf.placeholder(tf.float32, shape=(batch_size, horizon, weekly_len, num_nodes, int(input_dim/clip)), name='inputs_weekly')
        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, output_dim + aux_dim), name='labels')

        GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * (output_dim + aux_dim)))

        encgrucell = GRUCell(rnn_units, num_nodes)
        decgrucell = GRUCell(rnn_units, num_nodes)

        cell = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes, filter_type=filter_type)
        cell_with_projection = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                                         num_proj=output_dim, filter_type=filter_type)

        encoding_cells = [cell] * (num_rnn_layers)
        decoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]

        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('DCRNN_SEQ'):
            ### 1: clipped flow's cnn2
            #         self._inputs: [batch_size, seq_len, num_nodes, input_dim], input_dim = width_in * in_channels
            #         [batch_size * num_nodes, height_in=seq_len24, width_in, in_channels], Width_in=12, Channels_in=2
            #    CNN1 Filter_width=2, Channels_out=32, Width_out=6, width_stride=2
            #    CNN2 Filter_width=2, Channels_out=64, Width_out=5, width_stride=1
            # reshape input:
            inputs =  self._inputs
            # inputs = tf.reshape(tf.transpose(self._inputs, perm=[0, 2, 1, 3]), shape=[batch_size * num_nodes, seq_len, input_dim])
            # inputs = tf.transpose(self._inputs, perm=[0, 2, 1, 3])
            inputs = tf.expand_dims(inputs, axis=-1) #[batch_size, seq_len, num_nodes, input_dim, 1]
            dtype = inputs.dtype
            in_width = int(clip)
            in_channels = int(input_dim/clip)
            inputs = tf.reshape(inputs, shape=[batch_size, seq_len, num_nodes, in_channels, in_width])
            inputs = tf.transpose(inputs, perm=[0, 1, 2, 4, 3]) # [batch_size, seq_len, num_nodes, in_width, in_channels]
            # feed in No.1 CNN: 12 -> 6
            W1 = tf.get_variable('cnn1_w', [1, 1, 2, in_channels, 32], dtype=dtype,
                                 initializer=tf.contrib.layers.xavier_initializer())
            inputs = tf.nn.conv3d(inputs, W1, strides=[1, 1, 1, 2, 1], padding='VALID')
            b1 = tf.get_variable('cnn1_b', [32], dtype=dtype, initializer=tf.constant_initializer(0.0, dtype=dtype))
            inputs = cnn_activation(inputs + b1)
            # feed in No.2 CNN: 6 -> 5
            W2 = tf.get_variable('cnn2_w', [1, 1, 2, 32, 64], dtype=dtype,
                                 initializer=tf.contrib.layers.xavier_initializer())
            inputs = tf.nn.conv3d(inputs, W2, strides=[1, 1, 1, 1, 1], padding='VALID')
            b2 = tf.get_variable('cnn2_b', [64], dtype=dtype, initializer=tf.constant_initializer(0.0, dtype=dtype))
            inputs = cnn_activation(inputs + b2) # 320D
            # output_features transform:
            inputs = tf.reshape(inputs, shape=[batch_size, seq_len, num_nodes, -1])
            input_dim = inputs.get_shape()[-1].value

            ### 2: dayly and weekly flow's cnn2:
                # inputs_dayly:[batch_size, horizon, dayly_len, num_nodes, (output_dim + aux_dim)]
                # inputs_weekly:[batch_size, horizon, weekly_len, num_nodes, (output_dim + aux_dim)]
            # (1) reshape input:
            inputs_dayly = tf.transpose(self._inputs_dayly, perm=[0, 1, 3, 2, 4]) #[batch_size, horizon, num_nodes, dayly_len, (output_dim + aux_dim)]
            # inputs_dayly = tf.reshape(inputs_dayly, shape=[batch_size, horizon, num_nodes, -1])
            inputs_dayly = tf.unstack(inputs_dayly, axis=1) # horizon {[batch_size, num_nodes, dayly_len, (output_dim + aux_dim)]}
            inputs_weekly = tf.transpose(self._inputs_weekly, perm=[0, 1, 3, 2, 4])
            # inputs_dayly = tf.reshape(tf.concat([inputs_dayly, inputs_weekly], axis=3), shape=[batch_size, horizon, num_nodes, -1])
            # inputs_dayly = tf.unstack(inputs_dayly, axis=1)
            # inputs_weekly = tf.reshape(inputs_weekly, shape=[batch_size, horizon, num_nodes, -1])
            # inputs_dayweekly = tf.concat([inputs_dayly, inputs_weekly], axis=3)
            # inputs_dayweekly = tf.unstack(inputs_dayweekly, axis=1) # horizon {[batch_size, num_nodes, dayweekly_len, (output_dim + aux_dim)]]}
            inputs_weekly = tf.unstack(inputs_weekly, axis=1) # horizon {[batch_size, num_nodes, weekly_len, (output_dim + aux_dim)]}
            ### 1+2 -> 3: connect to DCRNN_seq2seq frame:
            inputs = tf.unstack(tf.reshape(inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)
            inputs_small = tf.unstack(tf.reshape(self._inputs_small, (batch_size, seq_len, -1)), axis=1)
            labels = tf.unstack(
                tf.reshape(self._labels[..., :output_dim], (batch_size, horizon, num_nodes * output_dim)), axis=1)
            if aux_dim > 0:
                aux_info = tf.unstack(self._labels[..., output_dim:], axis=1)
                aux_info.insert(0, None)
            labels.insert(0, GO_SYMBOL) # [batch_size, num_nodes * (output_dim + aux_dim)]

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
            with tf.variable_scope("rnn_encoder") as rnn_vs:
                state = encgrucell.zero_state(batch_size, dtype=tf.float32)
                state_small = encoding_cells[0].zero_state(batch_size, dtype=tf.float32)
                outputs = []
                for i, (inp, inp_small) in enumerate(zip(inputs, inputs_small)):
                    if i > 0:
                        rnn_vs.reuse_variables()

                    with tf.variable_scope('cell_0'):
                        output_small, state_small = encoding_cells[0](inp_small, state_small)
                    with tf.variable_scope('grucell'):
                        enc_outputs, state = encgrucell(inp, state)

                    # concat output_small, enc_outputs:
                    enc_outputs = tf.concat([output_small, enc_outputs], axis=-1)

                    for cell_i, cell in enumerate(encoding_cells):
                        if cell_i > 0:
                            with tf.variable_scope('cell_%d' % cell_i):
                                enc_outputs, state_small = cell(enc_outputs, state_small)
                    outputs.append(enc_outputs)
            # rnn_decoder:
            # inputs_dayly: horizon {[batch_size, num_nodes, dayly_len, (output_dim + aux_dim)]}
            # prev: [batch_size, num_nodes, output_dim]
            with tf.variable_scope("dayweeklycnn_rnn_decoder") as cnnrnn_vs:
                outputs = []
                prev = None
                for i, inp in enumerate(labels):
                    if i > 0:
                        cnnrnn_vs.reuse_variables()
                    if loop_function is not None and prev is not None:
                        with tf.variable_scope("loop_function", reuse=True):
                            inp = loop_function(prev, i)
                    if i < len(labels) - 1:
                        # (1) dayly cnn3:
                        #  concant dayly (output_dim + aux_dim):
                        dayly_inp = inputs_dayly[i]
                        # dayly_inp = tf.concat([inputs_dayly[i], inp], axis=2) # [batch_size, num_nodes, dayly_len+1, (output_dim + aux_dim)]
                        # dayly-cnn1: 12 -> 6
                        dayly_W1 = tf.get_variable('dayly_cnn1_w', [1, 2, (output_dim + aux_dim), 32], dtype=dtype, initializer=tf.contrib.layers.xavier_initializer())
                        dayly_inp = tf.nn.conv2d(dayly_inp, dayly_W1, strides=[1, 1, 2, 1], padding='VALID')
                        dayly_b1 = tf.get_variable('dayly_cnn1_b', [32], dtype=dtype, initializer=tf.constant_initializer(0.0, dtype=dtype))
                        dayly_inp = cnn_activation(dayly_inp + dayly_b1)
                        #       cnn2: 6 -> 5
                        # dayly_inp = tf.nn.avg_pool(dayly_inp, [1, 1, 5, 1], strides=[1, 1, 1, 1], padding='VALID')
                        dayly_W2 = tf.get_variable('dayly_cnn2_w', [1, 2, 32, 64], dtype=dtype, initializer=tf.contrib.layers.xavier_initializer())
                        dayly_inp = tf.nn.conv2d(dayly_inp, dayly_W2, strides=[1, 1, 1, 1], padding='VALID')
                        dayly_b2 = tf.get_variable('dayly_cnn2_b', [64], dtype=dtype, initializer=tf.constant_initializer(0.0, dtype=dtype))
                        dayly_inp = cnn_activation(dayly_inp + dayly_b2)
                        dayly_inp = tf.reshape(dayly_inp, shape=[batch_size, num_nodes, -1])   #320D
                        # (2) weekly cnn1:
                        #  concant weekly (output_dim + aux_dim):
                        weekly_inp = inputs_weekly[i]  # [batch_size, num_nodes, dayly_len, (output_dim + aux_dim)]
                        # weekly-cnn1: 4 -> 3
                        weekly_W1 = tf.get_variable('weekly_cnn1_w', [1, 2, (output_dim + aux_dim), 64], dtype=dtype, initializer=tf.contrib.layers.xavier_initializer())
                        weekly_inp = tf.nn.conv2d(weekly_inp, weekly_W1, strides=[1, 1, 1, 1], padding='VALID')
                        weekly_b1 = tf.get_variable('weekly_cnn1_b', [64], dtype=dtype, initializer=tf.constant_initializer(0.0, dtype=dtype))
                        weekly_inp = cnn_activation(weekly_inp + weekly_b1)
                        weekly_inp = tf.reshape(weekly_inp, shape=[batch_size, num_nodes, -1])   #192D
                        # (3) concat dayly_inp + weekly_inp:
                        dayweekly_inp = tf.concat([dayly_inp, weekly_inp], axis=2) # inp [batch_size, num_nodes, 512]
                        dayweekly_inp = tf.reshape(dayweekly_inp, shape=[batch_size, -1])

                        with tf.variable_scope('grucell'):
                            dec_outputs, state = decgrucell(dayweekly_inp, state)
                        with tf.variable_scope('cell_0'):
                            output_small, state_small = decoding_cells[0](inp, state_small)
                        # concat output_small, dec_output:
                        dec_outputs = tf.concat([output_small, dec_outputs], axis=-1)  # 128D

                        # to decoder rnn:
                        for cell_i, cell in enumerate(decoding_cells):
                            if cell_i > 0:
                                with tf.variable_scope('cell_%d' % cell_i):
                                    dec_outputs, state_small = cell(dec_outputs, state_small)
                        outputs.append(dec_outputs)
                        if loop_function is not None:
                            prev = dec_outputs
        # Project the output to output_dim.  outputs = tf.stack(outputs[:-1], axis=1)
        outputs = tf.stack(outputs, axis=1)
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
