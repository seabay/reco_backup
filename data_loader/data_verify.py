
import tensorflow as tf
tf.enable_eager_execution()

def parse_function(example_proto):
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


tfrecord_dataset = tf.data.TFRecordDataset(['../../data//validation/split_9/part-r-00000'])

parsed_dataset = tfrecord_dataset.map(parse_function)

for row in parsed_dataset:
    print(row[1])


