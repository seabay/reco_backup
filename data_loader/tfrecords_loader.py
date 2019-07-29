import sys
sys.path.append('..')

import tensorflow as tf
from configs import configs

class tfrecords_loader():

    def __init__(self, paths):
        self.paths = paths

    def parse_function(self, example_proto):
        features = {"seq_categorical_0":tf.FixedLenFeature([configs.max_transaction_history],tf.int64),
                    "seq_categorical_1":tf.FixedLenFeature([configs.max_product_click_history], tf.int64),
                    "seq_categorical_2":tf.FixedLenFeature([configs.max_promotion_click_history], tf.int64),
                    "105":tf.FixedLenFeature([1], tf.int64),
                    "106":tf.FixedLenFeature([1], tf.int64),
                    "107":tf.FixedLenFeature([1], tf.int64),
                    "numeric":tf.FixedLenFeature([10], tf.float32),
                    "labels":tf.FixedLenFeature([configs.category_size], tf.float32),
            }
        parsed_features = tf.parse_single_example(example_proto, features)
        #return parsed_features
        return (parsed_features["seq_categorical_0"], parsed_features["seq_categorical_1"], parsed_features["seq_categorical_2"], parsed_features["105"], parsed_features["106"], parsed_features["107"], parsed_features["numeric"]), parsed_features["labels"]


    def load(self):
        ds = tf.data.TFRecordDataset(self.paths)
        ds = ds.shuffle(buffer_size=10000)
        ds = ds.repeat()  # Repeat the input indefinitely.
        ds = ds.map(self.parse_function, num_parallel_calls=8)
        ds = ds.batch(configs.batch_size)
        ds = ds.prefetch(1)
        return ds

    

