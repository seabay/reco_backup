import tensorflow as tf

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))

def save_as_tfrecords(output_filename, features, labels):
    with tf.io.TFRecordWriter(output_filename) as writer:
        for (v1,v2,v3,v4,v5,v6, v7) in zip(features[0], features[1], features[2], features[3], features[4], features[5], labels):
            features = {'txSeq': int64_feature(v1), 'clickSeq': int64_feature(v2),
                        'genderIndex': int64_feature(v3),'is_email_verifiedIndex': int64_feature(v4), 'age': int64_feature(v5), 'cityIndex': int64_feature(v6), \
                            'label': int64_feature(v7)}
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())