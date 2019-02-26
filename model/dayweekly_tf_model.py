"""
Base class for tensorflow models for traffic forecasting.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class TFModel(object):
    def __init__(self, config, scaler=None, **kwargs):
        """
        Initialization including placeholders, learning rate,
        :param config:
        :param scaler: data z-norm normalizer
        :param kwargs:
        """
        self._config = dict(config)

        # Placeholders for input and output.
        self._inputs = None
        self._inputs_dayly = None
        self._inputs_weekly = None
        self._labels = None
        self._outputs = None

        # Scaler for data normalization.
        self._scaler = scaler

        # Train and loss
        self._loss = None
        self._mae = None
        self._train_op = None

        # Learning rate.
        learning_rate = config.get('learning_rate', 0.001)
        self._lr = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(learning_rate),
                                   trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr, name='lr_update')

        # Log merged summary
        self._merged = None


    def rnn_encoder_variant(self, inputs, batch_size, encoding_cells, initial_states=None):
        # rnn_encoder:
        with tf.variable_scope("rnn_encoder") as rnn_vs:
            if initial_states == None:
                state = encoding_cells[0].zero_state(batch_size, dtype=tf.float32)
            else:
                state = initial_states
            outputs = []
            for i, inp in enumerate(inputs):
                if i > 0:
                    rnn_vs.reuse_variables()
                for cell_i, cell in enumerate(encoding_cells):
                    with tf.variable_scope('cell_%d' % cell_i):
                        if cell_i == 0:
                            enc_outputs, state = cell(inp, state)
                        else:
                            enc_outputs, state = cell(enc_outputs, state)
                outputs.append(enc_outputs)
        return outputs, state

    def rnn_decoder_variant(self, labels, initial_states, decoding_cells, loop_function=None):
        # rnn_decoder:
        with tf.variable_scope("rnn_decoder") as rnn_vs:
            state = initial_states
            outputs = []
            prev = None
            for i, inp in enumerate(labels):
                if i > 0:
                    rnn_vs.reuse_variables()
                if loop_function is not None and prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        inp = loop_function(prev, i)
                for cell_i, cell in enumerate(decoding_cells):
                    with tf.variable_scope('cell_%d' % cell_i):
                        if cell_i == 0:
                            dec_outputs, state = cell(inp, state)
                        else:
                            dec_outputs, state = cell(dec_outputs, state)
                outputs.append(dec_outputs)
                if loop_function is not None:
                    prev = dec_outputs
        return outputs, state

    def attention_rnn_decoder_variant(self, labels, num_nodes, num_heads, initial_states, decoding_cells, loop_function=None, attention_function=None):
        # rnn_decoder:
        batch_size = labels[0].get_shape()[0].value
        # input_size = int(labels[0].get_shape()[1].value/num_nodes)
        rnn_units = int(decoding_cells[0].output_size/num_nodes)
        # if input_size < rnn_units * 2:
        #     input_size = rnn_units * 2  # 128d
        # output_size = int(decoding_cells[-1].output_size/num_nodes)
        with tf.variable_scope("rnn_decoder") as rnn_vs:
            state = initial_states
            outputs = []
            prev = None
            for i, inp in enumerate(labels):
                if i > 0:
                    rnn_vs.reuse_variables()
                if loop_function is not None and prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        inp = loop_function(prev, i)
                # Run the attention mechanism
                attns = attention_function([tf.reshape(state, shape=[batch_size * num_nodes, -1])])
                # Merge input and previous attentions into one vector of the right size.
                # inp = tf.reshape(inp, shape=[batch_size * num_nodes, -1])
                att_out = tf.reshape(tf.concat(attns, axis=1), shape=[batch_size, num_nodes, num_heads, -1])
                # conv1*1:
                w1 = tf.get_variable('cnn1_w', [1, num_heads, att_out.get_shape()[-1].value, rnn_units],
                                             dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                att_out = tf.nn.conv2d(att_out, w1, strides=[1, 1, 1, 1], padding='VALID')
                b1 = tf.get_variable('cnn1_b', [rnn_units], dtype=tf.float32,
                                             initializer=tf.constant_initializer(0.0, dtype=tf.float32))
                att_out = tf.reshape(tf.nn.relu(att_out + b1), shape=[batch_size, -1])
                inp = tf.concat([inp, att_out * state], axis=1)
                for i, cell in enumerate(decoding_cells):
                    with tf.variable_scope('cell_%d' % i):
                        if i == 0:
                            dec_outputs, state = cell(inp, state)
                        else:
                            dec_outputs, state = cell(dec_outputs, state)
                # with tf.variable_scope("AttnOutputProjection"):
                #     dec_outputs = tf.reshape(self.linear([tf.reshape(dec_outputs, shape=[batch_size * num_nodes, -1])] + attns, output_size), shape=[batch_size, -1])
                outputs.append(dec_outputs)
                if loop_function is not None:
                    prev = dec_outputs
        return outputs, state

    def linear(self, args, output_size, bias_start=0.0, scope=None, bias=True):
        """Graph linear of (list) input args.

        :param args: a 2D Tensor or a list of 2D, (batch*num_nodes) * input_dim, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            total_arg_size += shape[1].value
        dtype = [a.dtype for a in args][0]
        # Now the computation.
        iscope = tf.get_variable_scope()
        with tf.variable_scope(iscope):
            weights = tf.get_variable('weights', [total_arg_size, output_size], dtype=dtype, initializer=tf.contrib.layers.xavier_initializer())
            if len(args) == 1:
                res = tf.matmul(args[0], weights)
            else:
                res = tf.matmul(tf.concat(args, axis=1), weights)
            if bias:
                biases = tf.get_variable('biases', [output_size], dtype=dtype, initializer=tf.constant_initializer(bias_start, dtype=dtype))
                res = tf.nn.bias_add(res, biases)
        return res   # [batch_num_nodes_size, output_size]


    def rnn_encoder(self, inputs, batch_size, encoding_cells):
        # rnn_encoder:
        with tf.variable_scope('rnn_encoder') as enc_rnn_vs:
            enc_states = tuple(cell.zero_state(batch_size, dtype=tf.float32) for cell in encoding_cells)
            enc_outputs = []
            for enc_time, enc_input in enumerate(inputs):
                if enc_time > 0:
                    enc_rnn_vs.reuse_variables()
                enc_cur_inp = enc_input
                enc_new_states = []
                for enc_i, enc_cell in enumerate(encoding_cells):
                    with tf.variable_scope('cell_%d' % enc_i):
                        enc_cur_state = enc_states[enc_i]
                        enc_cur_inp, new_enc_state = enc_cell(enc_cur_inp, enc_cur_state)
                        enc_new_states.append(new_enc_state)
                enc_output = enc_cur_inp
                enc_states = enc_new_states = tuple(enc_new_states)
                enc_outputs.append(enc_output)
        return enc_outputs, enc_states

    def rnn_decoder(self, labels, initial_states, decoding_cells, loop_function=None):
        # rnn_decoder:
        with tf.variable_scope("rnn_decoder") as dec_rnn_vs:
            # state_c, _ = tf.split(state, num_or_size_splits=2, axis=-1)
            # decoder_state = tf.concat([enc_outputs[-1], enc_outputs[-1]], 1)
            dec_states = initial_states
            dec_outputs = []
            prev = None
            for dec_time, dec_input in enumerate(labels):
                if dec_time > 0:
                    dec_rnn_vs.reuse_variables()
                if loop_function is not None and prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        dec_input = loop_function(prev, dec_time)
                dec_cur_inp = dec_input
                dec_new_states = []
                for dec_i, dec_cell in enumerate(decoding_cells):
                    with tf.variable_scope('cell_%d' % dec_i):
                        dec_cur_state = dec_states[dec_i]
                        dec_cur_inp, new_dec_state = dec_cell(dec_cur_inp, dec_cur_state)
                        dec_new_states.append(new_dec_state)
                dec_output = dec_cur_inp
                dec_states = dec_new_states = tuple(dec_new_states)
                if loop_function is not None:
                    prev = dec_output
                dec_outputs.append(dec_output)
        return dec_outputs, dec_states

    def mlp_endecoder(self, labels, inputs, mlp_cells, loop_function=None):
        # inputs: (batch_size, num_nodes, input_dim)  labels: (batch_size, num_nodes, output_dim)
        # outputs: horizon (batch_size, num_nodes * output_dim)
        with tf.variable_scope("mlp_endecoder") as mlp_vs:
            batch_size = inputs[0].get_shape()[0].value
            num_nodes = inputs[0].get_shape()[1].value
            input_dim = inputs[0].get_shape()[2].value
            seq_len = len(inputs)
            outputs = []
            prev = None
            for time, input in enumerate(labels):
                if time > 0:
                    mlp_vs.reuse_variables()
                if loop_function is not None and prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        input = loop_function(prev, time)
                if time > 0:
                    inputs.append(input)
                    inputs = inputs[1:]
                # [batch_size, num_nodes, new_seq_len, input_dim] ->
                stack_inputs = tf.reshape(tf.stack(inputs, axis=2), shape=[batch_size, num_nodes, seq_len * input_dim])
                stack_inputs = tf.reshape(stack_inputs, shape=[batch_size, num_nodes * seq_len * input_dim])
                output = mlp_cells(stack_inputs)
                if loop_function is not None:
                    prev = tf.reshape(output, shape=[batch_size, num_nodes, -1])
                outputs.append(output)
        return outputs

    def rnn_decoder_psrnn(self, GO_SYMBOL, labels, initial_states, decoding_cells, loop_function=None):
        # rnn_decoder:
        with tf.variable_scope("rnn_decoder") as dec_rnn_vs:
            # state_c, _ = tf.split(state, num_or_size_splits=2, axis=-1)
            # decoder_state = tf.concat([enc_outputs[-1], enc_outputs[-1]], 1)
            dec_states = initial_states
            dec_outputs = []
            prev = None
            for dec_time, dec_input in enumerate(labels):
                if dec_time > 0:
                    dec_rnn_vs.reuse_variables()
                if loop_function is not None and prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        dec_input = loop_function(prev, dec_time)
                else:
                    dec_input = GO_SYMBOL
                dec_cur_inp = dec_input
                dec_new_states = []
                for dec_i, dec_cell in enumerate(decoding_cells):
                    with tf.variable_scope('cell_%d' % dec_i):
                        dec_cur_state = dec_states[dec_i]
                        dec_cur_inp, new_dec_state = dec_cell(dec_cur_inp, dec_cur_state)
                        dec_new_states.append(new_dec_state)
                dec_output = dec_cur_inp
                dec_states = dec_new_states = tuple(dec_new_states)
                if loop_function is not None:
                    prev = dec_output
                dec_outputs.append(dec_output)
        return dec_outputs, dec_states

    @staticmethod
    def run_epoch(sess, model, inputs, inputs_dayly, inputs_weekly, labels, return_output=False, train_op=None, writer=None):
        losses = []
        maes = []
        outputs = []

        fetches = {
            'mae': model.mae,
            'loss': model.loss,
            'global_step': tf.train.get_or_create_global_step()
        }
        if train_op:
            fetches.update({
                'train_op': train_op,
            })
            merged = model.merged
            if merged is not None:
                fetches.update({'merged': merged})

        if return_output:
            fetches.update({
                'outputs': model.outputs
            })

        for _, (x, x_dayly, x_weekly, y) in enumerate(zip(inputs, inputs_dayly, inputs_weekly, labels)):
            feed_dict = {
                model.inputs: x,
                model.inputs_dayly: x_dayly,
                model.inputs_weekly: x_weekly,
                model.labels: y,
            }

            vals = sess.run(fetches, feed_dict=feed_dict)

            losses.append(vals['loss'])
            maes.append(vals['mae'])
            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])
            if return_output:
                outputs.append(vals['outputs'])

        results = {
            'loss': np.mean(losses),
            'mae': np.mean(maes)
        }
        if return_output:
            results['outputs'] = outputs
        return results

    def get_lr(self, sess):
        return np.asscalar(sess.run(self._lr))

    def set_lr(self, sess, lr):
        sess.run(self._lr_update, feed_dict={
            self._new_lr: lr
        })

    @property
    def inputs(self):
        return self._inputs

    @property
    def inputs_dayly(self):
        return self._inputs_dayly

    @property
    def inputs_weekly(self):
        return self._inputs_weekly

    @property
    def labels(self):
        return self._labels

    @property
    def loss(self):
        return self._loss

    @property
    def lr(self):
        return self._lr

    @property
    def mae(self):
        return self._mae

    @property
    def merged(self):
        return self._merged

    @property
    def outputs(self):
        return self._outputs

    @property
    def train_op(self):
        return self._train_op
