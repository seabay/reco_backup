import sys
sys.path.append('..')

import tensorflow as tf

from models import dnn_estimator_model as dnn

from configs import configs
from utils import exp_data_util
from data_loader import tfrecords_saver, tfrecords_loader

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

## create experiment numpy data
features, labels = exp_data_util.create_data(data_size=configs.test_data_size, max_transaction_history=configs.max_transaction_history, \
    max_promotion_click_history=configs.max_promotion_click_history, numeric_size=configs.numeric_size, category_size=configs.category_size)

## save as tfrecord file, running once is ok
tfrecords_saver.save_as_tfrecords('../data/train/tf.tfrecord',features, labels)

if __name__ == '__main__':
    model = dnn.RecoEstimator(configs.max_transaction_history, configs.max_product_click_history, configs.max_promotion_click_history, category_size=configs.category_size,
                    numeric_features_size = configs.numeric_size, input_embedding_size = configs.input_embedding_size,
                    single_categorical_features = configs.single_category_cols).model

    model.train(input_fn=tfrecords_loader.tfrecords_loader('../data/train').load_func, steps=4000)
    model.evaluate(input_fn=tfrecords_loader.tfrecords_loader('../data/train').load_func, steps=configs.test_data_size//configs.batch_size)
