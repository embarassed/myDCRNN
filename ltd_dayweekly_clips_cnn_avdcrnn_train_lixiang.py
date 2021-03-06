from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
import yaml
import redis
from pandas import Series,DataFrame
from six.moves import xrange
import numpy as np


from lib import log_helper
from lib.dcrnn_utils import load_graph_data
from model.ltd_dayweekly_clips_cnn_avdcrnn_supervisor import LTDDCRNNSupervisor

# flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('sensor_ids_filename', 'data/sensor_graph/graph_sensor_ids_lixiang.txt','File containing sensor ids separated by comma.')
flags.DEFINE_integer('batch_size', -1, 'Batch size')
flags.DEFINE_integer('cl_decay_steps', -1,
                     'Parameter to control the decay speed of probability of feeding groundth instead of model output.')
flags.DEFINE_string('config_filename', 'data/model/dayweekly_dcrnn_config_lixiang.yaml', 'Configuration filename for restoring the model.')
flags.DEFINE_integer('epochs', -1, 'Maximum number of epochs to train.')
flags.DEFINE_string('filter_type', None, 'laplacian/random_walk/dual_random_walk.')
flags.DEFINE_string('graph_pkl_filename', 'data/sensor_graph/adj_mat_lixiang.pkl',
                    'Pickle file containing: sensor_ids, sensor_id_to_ind_map, dist_matrix')
flags.DEFINE_integer('horizon', -1, 'Maximum number of timestamps to prediction.')
flags.DEFINE_float('l1_decay', -1.0, 'L1 Regularization')
flags.DEFINE_float('lr_decay', -1.0, 'Learning rate decay.')
flags.DEFINE_integer('lr_decay_epoch', -1, 'The epoch that starting decaying the parameter.')
flags.DEFINE_integer('lr_decay_interval', -1, 'Interval beteween each deacy.')
flags.DEFINE_float('learning_rate', -1, 'Learning rate. -1: select by hyperopt tuning.')
flags.DEFINE_string('log_dir', None, 'Log directory for restoring the model from a checkpoint.')
flags.DEFINE_string('loss_func', None, 'MSE/MAPE/RMSE_MAPE: loss function.')
flags.DEFINE_float('min_learning_rate', -1, 'Minimum learning rate')
flags.DEFINE_integer('nb_weeks', 26, 'How many week\'s data should be used for train/test.')
flags.DEFINE_integer('patience', 20,
                     'Maximum number of epochs allowed for non-improving validation error before early stopping.')
flags.DEFINE_integer('seq_len', 288, 'Sequence length. default: one day 288')
flags.DEFINE_integer('test_every_n_epochs', -1, 'Run model on the testing dataset every n epochs.')
flags.DEFINE_bool('use_cpu_only', False, 'Set to true to only use cpu.')
flags.DEFINE_bool('use_curriculum_learning', None, 'Set to true to use Curriculum learning in decoding stage.')
flags.DEFINE_integer('verbose', -1, '1: to log individual sensor information.')
flags.DEFINE_integer('max_diffusion_step', -1, 'maximum diffusion steps of [dual]random walk')
# flags.DEFINE_string('data', 'flow,flow_day_1,flow_day_2,flow_day_3,flow_day_4,flow_day_5,flow_day_6,flow_day_7,flow_day_8,flow_day_9,flow_day_10,flow_day_11,flow_day_12,flow_week_1,flow_week_2,flow_week_3,flow_week_4', 'input data type: speed/occ/flow')
flags.DEFINE_integer('clip', 12, 'clip number of seq_len (base_seqlen:24) >= 2')
flags.DEFINE_integer('dayly_len', 12, 'dayly_len')
flags.DEFINE_integer('weekly_len', 4, 'weekly_len')


def generateDataframeFromRedis(data):
    r = redis.Redis(host='172.17.0.1', port=6379, db=0) #172.17.0.1 server; 127.0.0.1 self notebook
    data_r = r.hgetall(data)
    with open(FLAGS.sensor_ids_filename) as f:
        sensor_ids = f.read().strip().split(',')
    ind = bytes.decode(data_r[b'time']).strip().split(',')
    for i in xrange(len(ind)):
        ind[i] = np.datetime64(ind[i])
    contents = {}
    for station in sensor_ids:
        contents[station] = bytes.decode(data_r[str.encode(station)]).strip().split(',')
        count = len(contents[station])
        for i in xrange(count):
            contents[station][i] = float(contents[station][i])
    return DataFrame(contents, index=ind)


def main():
    # Reads graph data.
    with open(FLAGS.config_filename) as f:
        supervisor_config = yaml.load(f)
        logger = log_helper.get_logger(supervisor_config.get('base_dir'), 'info.log')
        logger.info('Loading graph from: ' + FLAGS.graph_pkl_filename)
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(FLAGS.graph_pkl_filename)
        adj_mx[adj_mx < 0.1] = 0
        logger.info('Loading traffic data from Redis')
        traffic_reading_df_s = []
        # data_s = FLAGS.data.split(',')
        data_s = 'flow,flow_day_1,flow_day_2,flow_day_3,flow_day_4,flow_day_5,flow_day_6,flow_day_7,flow_day_8,flow_day_9,flow_day_10,flow_day_11,flow_day_12,flow_week_1,flow_week_2,flow_week_3,flow_week_4'.split(',')
        for i in range(len(data_s)):
            traffic_reading_df = generateDataframeFromRedis(data_s[i])
            traffic_reading_df = traffic_reading_df.ix[:, sensor_ids]
            print(traffic_reading_df.shape[0])
            traffic_reading_df_s.append(traffic_reading_df)
        if len(traffic_reading_df_s) != 17:
            print('failed to import data from redis!')
            return
        supervisor_config['use_cpu_only'] = FLAGS.use_cpu_only
        if FLAGS.log_dir:
            supervisor_config['log_dir'] = FLAGS.log_dir
        if FLAGS.use_curriculum_learning is not None:
            supervisor_config['use_curriculum_learning'] = FLAGS.use_curriculum_learning
        if FLAGS.loss_func:
            supervisor_config['loss_func'] = FLAGS.loss_func
        if FLAGS.filter_type:
            supervisor_config['filter_type'] = FLAGS.filter_type
        if FLAGS.clip:
            supervisor_config['clip'] = FLAGS.clip
        # Overwrites space with specified parameters.
        for name in ['batch_size', 'cl_decay_steps', 'epochs', 'horizon', 'learning_rate', 'l1_decay',
                     'lr_decay', 'lr_decay_epoch', 'lr_decay_interval', 'learning_rate', 'min_learning_rate',
                     'patience', 'seq_len', 'verbose', 'max_diffusion_step']:
            if getattr(FLAGS, name) >= 0:
                supervisor_config[name] = getattr(FLAGS, name)

        tf_config = tf.ConfigProto()
        if FLAGS.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = LTDDCRNNSupervisor(traffic_reading_df=traffic_reading_df_s[0],
                                            traffic_reading_df_dayly=traffic_reading_df_s[1:13],
                                            traffic_reading_df_weekly=traffic_reading_df_s[13:],
                                            adj_mx=adj_mx, config=supervisor_config)

            supervisor.train(sess=sess)


if __name__ == '__main__':
    main()
