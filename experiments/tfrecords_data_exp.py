import sys
sys.path.append('..')

import tensorflow as tf

import models.simple_dnn_model as model
from configs import configs
from utils import exp_data_util
from data_loader import tfrecords_saver, tfrecords_loader

single_category_cols = {105:3,106:5,107:10}   ## such as location : unique_value_size

## create experiment numpy data
features, labels = exp_data_util.create_data()

## save as tfrecord file
tfrecords_saver.save_as_tfrecords('../data/train-tf.tfrecord',features, labels)

## load tfrecords file
ds = tfrecords_loader.tfrecords_loader('../data', 'train').load()
#iter = ds.make_one_shot_iterator()

if __name__ == '__main__':

    model = model.RecoDNN(configs.max_transaction_history, configs.max_product_click_history, configs.max_promotion_click_history, configs.category_size,
                    numeric_features_size = configs.numeric_size, input_embedding_size = configs.input_embedding_size,
                    single_categorical_features = single_category_cols).model
    model.summary()

    #model.fit(iter, epochs=configs.epochs, steps_per_epoch=int(10000//configs.batch_size))
    model.fit(ds, epochs=configs.epochs, steps_per_epoch=int(10000//configs.batch_size))