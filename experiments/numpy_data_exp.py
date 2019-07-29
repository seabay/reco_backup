import sys
sys.path.append('..')

import tensorflow as tf

import models.simple_dnn_model as model
from configs import configs
from utils import exp_data_util

single_category_cols = {105:3,106:5,107:10}   ## such as location : unique_value_size

## create experiment tfrecord data
features, labels = exp_data_util.create_data()

if __name__ == '__main__':
    model = model.RecoDNN(configs.max_transaction_history, configs.max_product_click_history, configs.max_promotion_click_history, configs.category_size,
                    numeric_features_size = configs.numeric_size, input_embedding_size = configs.input_embedding_size,
                    single_categorical_features = single_category_cols).model

    model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

    model.summary()

    model.fit(x=features, y=tf.keras.utils.to_categorical(labels, num_classes=configs.category_size), epochs=configs.epochs, batch_size=configs.batch_size)


