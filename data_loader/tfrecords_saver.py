import tensorflow as tf

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))

def save_as_tfrecords(output_filename, features, one_hot_labels):
    with tf.io.TFRecordWriter(output_filename) as writer:
        for (v1,v2,v3,v4,v5,v6,v7,v8) in zip(features[0], features[1], features[2], features[3], features[4], features[5], features[6], one_hot_labels):
            features = {'seq_categorical_0': int64_feature(v1), 'seq_categorical_1': int64_feature(v2),
                        'seq_categorical_2': int64_feature(v3),'105': int64_feature(v4),
                        '106': int64_feature(v5),'107': int64_feature(v6), 'numeric': float_feature(v7),
                        'labels': float_feature(v8)}
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())