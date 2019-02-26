from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os
import sys
import tensorflow as tf
import time
import yaml
import pandas as pd

from lib import log_helper
from lib import metrics
from lib import tf_utils
from lib import utils
from lib.utils import StandardScaler
from model.tf_model import TFModel


class LTDTFModelFOSSupervisor(object):
    """
    Base supervisor for tensorflow models for traffic forecasting.
    """

    def __init__(self, config, df_data, df_addition_data_s, **kwargs):
        self._config = dict(config)
        self._epoch = 0

        # logging.
        self._init_logging()
        self._logger.info(config)

        # Data preparation
        test_ratio = self._get_config('test_ratio')
        validation_ratio = self._get_config('validation_ratio')
        self._df_train, self._df_val, self._df_test = utils.ltd_train_val_test_split_df(df_data, self._get_config('seq_len'),
                                                                                        val_ratio=validation_ratio,
                                                                                    test_ratio=test_ratio)
        self._scaler = StandardScaler(mean=self._df_train.values.mean(), std=self._df_train.values.std())
        self._x_train, self._y_train, self._x_val, self._y_val, self._x_test, self._y_test = self._prepare_train_val_test_data()
        self._addition_scaler_s, _addition_x_train_s, _addition_x_val_s, _addition_x_test_s = self._prepare_scaler_train_val_test_addition_data(df_addition_data_s)
        # self._addition_scaler_s should needn't use
        for addition_x_train in _addition_x_train_s:
            x_train_s = [self._x_train, addition_x_train]
            self._x_train = np.concatenate(x_train_s, axis=-1)
        for addition_x_val in _addition_x_val_s:
            x_valid_s = [self._x_val, addition_x_val]
            self._x_val = np.concatenate(x_valid_s, axis=-1)
        for addition_x_test in _addition_x_test_s:
            x_test_x = [self._x_test, addition_x_test]
            self._x_test = np.concatenate(x_test_x, axis=-1)
        self._eval_dfs = self._prepare_eval_df()

        # Build models.
        self._train_model, self._val_model, self._test_model = self._build_train_val_test_models()

        # Log model statistics.
        total_trainable_parameter = tf_utils.get_total_trainable_parameter_size()
        self._logger.info('Total number of trainable parameters: %d' % total_trainable_parameter)
        for var in tf.global_variables():
            self._logger.debug('%s, %s' % (var.name, var.get_shape()))

    def _get_config(self, key, use_default=True):
        default_config = {
            'add_day_in_week': False,
            'add_time_in_day': True,
            'dropout': 0.,
            'batch_size': 64,
            'horizon': 12,
            'learning_rate': 1e-3,
            'lr_decay': 0.1,
            'lr_decay_epoch': 50,
            'lr_decay_interval': 10,
            'max_to_keep': 100,
            'min_learning_rate': 2e-6,
            'null_val': 0.,
            'output_type': 'range',
            'patience': 20,
            'save_model': 1,
            'seq_len': 12,
            'test_batch_size': 1,
            'test_every_n_epochs': 10,
            'test_ratio': 0.2,
            'use_cpu_only': False,
            'validation_ratio': 0.1,
            'verbose': 0,
        }
        value = self._config.get(key)
        if value is None and use_default:
            value = default_config.get(key)
        return value

    def _init_logging(self):
        base_dir = self._get_config('base_dir')
        log_dir = self._get_config('log_dir')
        if log_dir is None:
            run_id = self._generate_run_id(self._config)
            log_dir = os.path.join(base_dir, run_id)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        else:
            run_id = os.path.basename(os.path.normpath(log_dir))
        # TO DO:
        self.run_id = run_id
        # TO DO END
        self._log_dir = log_dir
        self._logger = log_helper.get_logger(self._log_dir, run_id)
        self._writer = tf.summary.FileWriter(self._log_dir)

    def train(self, sess, **kwargs):
        history = []
        min_val_loss = float('inf')
        wait = 0

        epochs = self._get_config('epochs')
        initial_lr = self._get_config('learning_rate')
        min_learning_rate = self._get_config('min_learning_rate')
        lr_decay_epoch = self._get_config('lr_decay_epoch')
        lr_decay = self._get_config('lr_decay')
        lr_decay_interval = self._get_config('lr_decay_interval')
        patience = self._get_config('patience')

        sess.run(tf.global_variables_initializer())

        while self._epoch <= epochs:
            # Learning rate schedule.
            new_lr = self.calculate_scheduled_lr(initial_lr, epoch=self._epoch,
                                                 lr_decay=lr_decay, lr_decay_epoch=lr_decay_epoch,
                                                 lr_decay_interval=lr_decay_interval,
                                                 min_lr=min_learning_rate)
            if new_lr != initial_lr:
                self._logger.info('Updating learning rate to: %.6f' % new_lr)
                self._train_model.set_lr(sess=sess, lr=new_lr)
            sys.stdout.flush()

            start_time = time.time()
            train_results = TFModel.run_epoch(sess, self._train_model,
                                              inputs=self._x_train, labels=self._y_train,
                                              train_op=self._train_model.train_op, writer=self._writer)
            train_loss, train_mae = train_results['loss'], train_results['mae']
            if train_loss > 1e5:
                self._logger.warn('Gradient explosion detected. Ending...')
                break

            global_step = sess.run(tf.train.get_or_create_global_step())
            # Compute validation error.
            val_results = TFModel.run_epoch(sess, self._val_model, inputs=self._x_val, labels=self._y_val,
                                            train_op=None)
            val_loss, val_mae = val_results['loss'], val_results['mae']

            tf_utils.add_simple_summary(self._writer,
                                        ['loss/train_loss', 'metric/train_mae', 'loss/val_loss', 'metric/val_mae'],
                                        [train_loss, train_mae, val_loss, val_mae], global_step=global_step)
            end_time = time.time()
            message = 'Epoch %d (%d) train_loss: %.4f, train_mae: %.4f, val_loss: %.4f, val_mae: %.4f %ds' % (
                self._epoch, global_step, train_loss, train_mae, val_loss, val_mae, (end_time - start_time))
            self._logger.info(message)
            # if self._epoch % test_every_n_epochs == test_every_n_epochs - 1:
            #     self.test_and_write_result(sess=sess, global_step=global_step, epoch=self._epoch)

            if val_loss <= min_val_loss:
                wait = 0
                # if save_model > 0:
                    # TO DO:
                    # model_filename = self.save_model_lixiang(sess, saver, global_step, val_loss)
                    # model_filename = self.save_model(sess, saver, val_loss)
                #     TO DO END.
                # self._logger.info(
                #     'Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss, val_loss, model_filename))
                self._logger.info('Val loss decrease from %.4f to %.4f' % (min_val_loss, val_loss))
                # TO DO:
                if self._epoch >= 40:
                    df_preds_ = self.test_and_write_result(sess=sess, global_step=global_step, epoch=self._epoch)
                    result_dir = os.path.join('data/results/', self.run_id)
                    print(result_dir)
                    os.makedirs(result_dir,exist_ok=True)
                    for horizon_i_ in df_preds_:
                        df_pred_ = df_preds_[horizon_i_]
                        filename_ = os.path.join(result_dir, 'dcrnn_prediction_%d_%d.h5' % (self._epoch, horizon_i_ + 1))
                        # print(filename_, 'len of df_pred:', len(df_pred_.index.values))
                        df_pred_.to_hdf(filename_, 'results')
                    # print('Predictions saved as data/results/%s/dcrnn_seq2seq_prediction_%d_[1-12].h5...' % (self.run_id, self._epoch))
                # TO DO END.
                min_val_loss = val_loss
            else:
                wait += 1
                if wait > patience:
                    self._logger.warn('Early stopping at epoch: %d' % self._epoch)
                    break

            history.append(val_mae)
            # Increases epoch.
            self._epoch += 1

            sys.stdout.flush()
        return np.min(history)

    @staticmethod
    def calculate_scheduled_lr(initial_lr, epoch, lr_decay, lr_decay_epoch, lr_decay_interval,
                               min_lr=1e-6):
        decay_factor = int(math.ceil((epoch - lr_decay_epoch) / float(lr_decay_interval)))
        new_lr = initial_lr * lr_decay ** max(0, decay_factor)
        new_lr = max(min_lr, new_lr)
        return new_lr

    @staticmethod
    def _generate_run_id(config):
        raise NotImplementedError

    @staticmethod
    def _get_config_filename(epoch):
        return 'config_%02d.yaml' % epoch

    def test_and_write_result(self, sess, global_step, **kwargs):
        null_val = self._config.get('null_val')
        start_time = time.time()
        test_results = TFModel.run_epoch(sess, self._test_model, self._x_test, self._y_test, return_output=True,
                                         train_op=None)

        # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
        test_loss, y_preds = test_results['loss'], test_results['outputs']
        tf_utils.add_simple_summary(self._writer, ['loss/test_loss'], [test_loss], global_step=global_step)

        # Reshapes to (batch_size, epoch_size, horizon, num_node)
        df_preds = self._convert_model_outputs_to_eval_df(y_preds)

        for horizon_i in df_preds:
            df_pred = df_preds[horizon_i]
            df_test = self._eval_dfs[horizon_i]
            mae, mape, rmse = metrics.calculate_metrics(df_pred, df_test, null_val)

            tf_utils.add_simple_summary(self._writer,
                                        ['%s_%d' % (item, horizon_i + 1) for item in
                                         ['metric/rmse', 'metric/mape', 'metric/mae']],
                                        [rmse, mape, mae],
                                        global_step=global_step)
            end_time = time.time()
            message = 'Horizon %d, mape:%.4f, rmse:%.4f, mae:%.4f, %ds' % (
                horizon_i + 1, mape, rmse, mae, end_time - start_time)
            self._logger.info(message)
            start_time = end_time
        return df_preds

    def _prepare_train_val_test_data(self):
        """
        Prepare data for train, val and test.
        :return:
        """
        raise NotImplementedError

    def _prepare_scaler_train_val_test_addition_data(self, df_addition_data_s):
        """
        Prepare data addition for train, val and test, and these scaler_s
        :param df_addition_data_s:
        :return: addition_scaler_s, addition_x_train_s, addition_x_val_s, addition_x_test_s
        """
        addition_scaler_s = []
        addition_x_train_s = []
        addition_x_val_s = []
        addition_x_test_s = []
        addition_num = len(df_addition_data_s)
        batch_size = self._get_config('batch_size')
        horizon = self._get_config('horizon')
        seq_len = self._get_config('seq_len')
        test_batch_size = 1
        for ind in range(addition_num):
            df = df_addition_data_s[ind]
            n_sample, _ = df.shape
            n_val = int(round(n_sample * self._get_config('validation_ratio')))
            n_test = int(round(n_sample * self._get_config('test_ratio')))
            n_train = n_sample - n_val - n_test
            train_data, val_data, test_data = df.iloc[:n_train, :], df.iloc[n_train - seq_len: n_train + n_val, :], df.iloc[-n_test-seq_len:, :]
            scaler = StandardScaler(mean=train_data.values.mean(), std=train_data.values.std())
            num_nodes = train_data.shape[-1]
            x_train, _ = utils.ltd_generate_graph_seq2seq_io_data_with_time(train_data,
                                                                  batch_size=batch_size,
                                                                  seq_len=seq_len,
                                                                  horizon=horizon,
                                                                  num_nodes=num_nodes,
                                                                  scaler=scaler,
                                                                  add_time_in_day=False,
                                                                  add_day_in_week=False)
            x_val, _ = utils.ltd_generate_graph_seq2seq_io_data_with_time(val_data, batch_size=batch_size,
                                                                    seq_len=seq_len,
                                                                    horizon=horizon,
                                                                    num_nodes=num_nodes,
                                                                    scaler=scaler,
                                                                    add_time_in_day=False,
                                                                    add_day_in_week=False)
            x_test, _ = utils.ltd_generate_graph_seq2seq_io_data_with_time(test_data,
                                                                      batch_size=test_batch_size,
                                                                      seq_len=seq_len,
                                                                      horizon=horizon,
                                                                      num_nodes=num_nodes,
                                                                      scaler=scaler,
                                                                      add_time_in_day=False,
                                                                      add_day_in_week=False)
            addition_scaler_s.append(scaler)
            addition_x_train_s.append(x_train)
            addition_x_val_s.append(x_val)
            addition_x_test_s.append(x_test)
        return addition_scaler_s, addition_x_train_s, addition_x_val_s, addition_x_test_s

    def _prepare_eval_df(self):
        horizon = self._get_config('horizon')
        seq_len = self._get_config('seq_len')
        # y_test: (epoch_size, batch_size, ...)
        n_test_samples = np.prod(self._y_test.shape[:2])
        eval_dfs = {}
        for horizon_i in range(horizon):
            eval_dfs[horizon_i] = self._df_test[seq_len + horizon_i: seq_len + horizon_i + n_test_samples]
        return eval_dfs

    def _build_train_val_test_models(self):
        """
        Buids models for train, val and test.
        :return:
        """
        raise NotImplementedError

    def _convert_model_outputs_to_eval_df(self, y_preds):
        """
        Convert the outputs to a dict, with key: horizon, value: the corresponding dataframe.
        :param y_preds:
        :return:
        """
        raise NotImplementedError

    @property
    def log_dir(self):
        return self._log_dir
