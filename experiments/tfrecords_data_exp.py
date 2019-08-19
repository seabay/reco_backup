import sys
sys.path.append('..')

import tensorflow as tf

from models import simple_dnn_model as dnn
from models import simple_cnn_model as cnn

from configs import configs
from utils import exp_data_util
from data_loader import tfrecords_saver, tfrecords_loader

tf.compat.v1.logging.set_verbosity(tf.logging.INFO)

## create experiment numpy data
features, labels = exp_data_util.create_data(data_size=configs.test_data_size, max_transaction_history=configs.max_transaction_history, \
    max_promotion_click_history=configs.max_promotion_click_history, numeric_size=configs.numeric_size, category_size=configs.category_size)

## save as tfrecord file, running once is ok
tfrecords_saver.save_as_tfrecords('../data/train/tf.tfrecord',features, labels)

## load tfrecords file
ds = tfrecords_loader.tfrecords_loader('../data/train').load()
#iter = ds.make_one_shot_iterator()

if __name__ == '__main__':
    model = dnn.RecoDNN(configs.max_transaction_history, configs.max_product_click_history, configs.max_promotion_click_history, category_size=configs.category_size,
                    numeric_features_size = configs.numeric_size, input_embedding_size = configs.input_embedding_size,
                    single_categorical_features = configs.single_category_cols).model

    model.summary()

    tf.keras.utils.plot_model(model, to_file='../figures/model.png', show_shapes=True, show_layer_names=True)

    #model_est=tf.keras.estimator.model_to_estimator(keras_model=model, model_dir="kkt")
    #train_input = lambda:tfrecords_loader.tfrecords_loader('../data/train').load_func()
    #model_est.train(input_fn=train_input, steps=300)
    model.fit(ds, epochs=50, steps_per_epoch=int(configs.test_data_size//configs.batch_size))
