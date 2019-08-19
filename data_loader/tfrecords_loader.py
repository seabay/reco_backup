import sys
sys.path.append('..')

import os
import tensorflow as tf
from configs import configs

class tfrecords_loader():

    def __init__(self, path):
        self.path = path
        self.files = self.getListOfFiles(self.path)
        print('loading files: ', self.files)

    def parse_function(self, example_proto):
        features = {"txSeq":tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    "clickSeq":tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    "genderIndex":tf.io.FixedLenFeature([1], tf.int64),
                    "is_email_verifiedIndex":tf.io.FixedLenFeature([1], tf.int64),
                    "age":tf.io.FixedLenFeature([1], tf.int64),
                    "cityIndex":tf.io.FixedLenFeature([1], tf.int64),
                    "label":tf.io.FixedLenFeature([1], tf.int64)
                    }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        return (parsed_features["txSeq"], parsed_features["clickSeq"], parsed_features["genderIndex"], parsed_features["is_email_verifiedIndex"], parsed_features["age"], parsed_features["cityIndex"]), parsed_features["label"]
        
        #return (parsed_features["txSeq"], parsed_features["clickSeq"]), parsed_features["label"]

    def parse_function_as_dict(self, example_proto):
        features = {"txSeq":tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    "clickSeq":tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    "genderIndex":tf.io.FixedLenFeature([1], tf.int64),
                    "is_email_verifiedIndex":tf.io.FixedLenFeature([1], tf.int64),
                    "age":tf.io.FixedLenFeature([1], tf.int64),
                    "cityIndex":tf.io.FixedLenFeature([1], tf.int64),
                    "label":tf.io.FixedLenFeature([1], tf.int64)
                    }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        return {'txSeq':parsed_features["txSeq"], 'clickSeq':parsed_features["clickSeq"], 'genderIndex':parsed_features["genderIndex"], 'is_email_verifiedIndex':parsed_features["is_email_verifiedIndex"], 'age':parsed_features["age"], 'cityIndex':parsed_features["cityIndex"]}, parsed_features["label"]
        

    def load(self):
        padded_shapes = (([configs.max_transaction_history], [configs.max_promotion_click_history], [None],[None],[None], [None]), [None])
        #padded_shapes = (([configs.max_transaction_history], [configs.max_promotion_click_history]), [None])
        ds = tf.data.TFRecordDataset(self.files, buffer_size=configs.buffer_size, num_parallel_reads=os.cpu_count())
        ds = ds.map(self.parse_function, num_parallel_calls=os.cpu_count())
        ds = ds.shuffle(buffer_size=configs.buffer_size)
        ds = ds.padded_batch(configs.batch_size, padded_shapes)
        ds = ds.prefetch(2)
        ds = ds.repeat()
        return ds

    def load_func(self):
        #padded_shapes = (([configs.max_transaction_history], [configs.max_promotion_click_history], [None],[None],[None], [None]), [None])
        #padded_shapes = (([configs.max_transaction_history], [configs.max_promotion_click_history]), [None])
        padded_shapes = ({'txSeq':[configs.max_transaction_history], 'clickSeq':[configs.max_promotion_click_history], 'genderIndex':[None], 'is_email_verifiedIndex':[None],
        'age':[None], 'cityIndex':[None]}, [None])
        ds = tf.data.TFRecordDataset(self.files,buffer_size=configs.buffer_size, num_parallel_reads=os.cpu_count())
        ds = ds.map(self.parse_function_as_dict, num_parallel_calls=os.cpu_count())
        ds = ds.shuffle(buffer_size=configs.buffer_size)
        ds = ds.padded_batch(configs.batch_size, padded_shapes)
        ds = ds.prefetch(1)
        ds = ds.repeat()
        #iterator = ds.make_one_shot_iterator()
        #features, labels = iterator.get_next()
        #return features, labels
        return ds

    def load2(self):
        padded_shapes = (([configs.max_transaction_history], [configs.max_promotion_click_history], [None],[None],[None], [None]), [None])
        ds = (tf.data.Dataset.from_tensor_slices(self.files).interleave(lambda x: tf.data.TFRecordDataset(x).map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE),cycle_length=4, block_length=32))
        ds = ds.padded_batch(configs.batch_size, padded_shapes)
        ds = ds.prefetch(8)
        return ds


    def getListOfFiles(self, dirName):
        
        listOfFile = os.listdir(dirName)
        allFiles = list()
        
        for entry in listOfFile:
            fullPath = os.path.join(dirName, entry)
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)
                    
        return allFiles



