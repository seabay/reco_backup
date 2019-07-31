import sys
sys.path.append('..')

import tensorflow as tf
from configs import configs
import os

class tfrecords_loader():

    def __init__(self, tfrecords_dir, subset='train'):
        self.tfrecords_dir = tfrecords_dir
        self.subset = subset

    def parse_function(self, example_proto):
        features = {"seq_categorical_0":tf.io.FixedLenFeature([configs.max_transaction_history],tf.int64),
                    "seq_categorical_1":tf.io.FixedLenFeature([configs.max_product_click_history], tf.int64),
                    "seq_categorical_2":tf.io.FixedLenFeature([configs.max_promotion_click_history], tf.int64),
                    "105":tf.io.FixedLenFeature([1], tf.int64),
                    "106":tf.io.FixedLenFeature([1], tf.int64),
                    "107":tf.io.FixedLenFeature([1], tf.int64),
                    "numeric":tf.io.FixedLenFeature([10], tf.float32),
                    "labels":tf.io.FixedLenFeature([1], tf.float32),
            }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        #return parsed_features
        return (parsed_features["seq_categorical_0"], parsed_features["seq_categorical_1"], parsed_features["seq_categorical_2"], parsed_features["105"], parsed_features["106"], parsed_features["107"], parsed_features["numeric"]), parsed_features["labels"]

    def parse_function_valid(self):
        pass

    def load(self):
        files = tf.io.matching_files(os.path.join(self.tfrecords_dir, '%s-*' % self.subset))
        ds = tf.data.TFRecordDataset(files)
        ds = ds.shuffle(buffer_size=10000)
        ds = ds.repeat()  # Repeat the input indefinitely.
        ds = ds.map(self.parse_function, num_parallel_calls=8)
        ds = ds.batch(configs.batch_size)
        ds = ds.prefetch(1)
        return ds

    def load2(self):
        """Read TFRecords files and turn them into a TFRecordDataset."""
        files = tf.io.matching_files(os.path.join(self.tfrecords_dir, '%s-*' % self.subset))
        shards = tf.data.Dataset.from_tensor_slices(files)
        shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
        shards = shards.repeat()
        dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
        dataset = dataset.shuffle(buffer_size=8192)
        parser = self.parse_function if self.subset == 'train' else self.parse_function_valid
        dataset = dataset.map(parser, num_parallel_calls=configs.num_parallel_calls)
        dataset = dataset.batch(configs.batch_size)
        dataset = dataset.prefetch(1)
        return dataset
    

