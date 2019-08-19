import numpy as np
import builtins
import tensorflow as tf


class RecoCNN():
    
    def __init__(self, max_transaction_history = 50, max_product_click_history = 50, max_promotion_click_history = 50,
                 category_size = 100, single_categorical_features = None, numeric_features_size = 10,
                 hidden_layer1_size = 256, hidden_layer2_size = 128, hidden_layer3_size = 64, activation='relu',
                input_embedding_size = 64, multi_gpu_model=False):
        
        self.max_transaction_history = max_transaction_history
        self.max_product_click_history = max_product_click_history
        self.max_promotion_click_history = max_promotion_click_history
        self.category_size = category_size
        self.hidden_layer1_size = hidden_layer1_size
        self.hidden_layer2_size = hidden_layer2_size
        self.hidden_layer3_size = hidden_layer3_size
        self.single_categorical_features = single_categorical_features
        self.numeric_features_size = numeric_features_size
        self.activation = activation
        self.input_embedding_size = input_embedding_size
        self.multi_gpu_model = multi_gpu_model
        
        self.category_embeddings = tf.keras.layers.Embedding(output_dim=self.input_embedding_size, 
                                                             input_dim = self.category_size, mask_zero=False, name='category_embeddings')
        self.filter_sizes = [2,3,4,5]
        self.num_filters = 512
        
        self.dropout = 0.2

        self.model = None
        self.build()
        
    
    def build(self):
        seq_layer, seq_embed, singles = self.create_input()
        flatten = self.cnn_seq_encode(seq_embed)
        flatten = tf.keras.layers.Dropout(self.dropout)(flatten)
        merge_input = self.merge_seq_single(flatten, singles)
        v = tf.keras.layers.Dense(512, activation = self.activation)(merge_input)
        v = tf.keras.layers.LayerNormalization()(v)
        v = tf.keras.layers.Dense(self.hidden_layer1_size, activation = self.activation)(v)
        v = tf.keras.layers.LayerNormalization()(v)
        v = tf.keras.layers.Dense(self.hidden_layer2_size, activation = self.activation)(v)
        v = tf.keras.layers.LayerNormalization()(v)
        v = tf.keras.layers.Dense(self.hidden_layer3_size, activation = self.activation, name='user_embedding')(v)
        v = tf.keras.layers.LayerNormalization()(v)
        output = tf.keras.layers.Dense(self.category_size, activation ='softmax', name='softmax_layer')(v)
        self.model = tf.keras.models.Model(inputs = seq_layer + [s[0] for s in singles], outputs = [output])  

        if self.multi_gpu_model:
            
            try:
                self.model = tf.keras.utils.multi_gpu_model(self.model, gpus=8, cpu_relocation=True)
                print("Training using multiple GPUs..")
            except:
                print("Training using single GPU or CPU..")
            

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy']) 
        
    def merge_seq_single(self, flatten, singles):
        cat_ = [flatten]
        cat_ += [s[1] for s in singles]
        return tf.keras.layers.concatenate(cat_, axis=1)
    
    def cnn_seq_encode(self, seq_embed):
        
        cat_embedding = tf.keras.layers.concatenate(seq_embed, axis=1)
        cat_embedding = tf.keras.layers.Reshape((self.max_transaction_history*2, self.input_embedding_size,1))(cat_embedding)
        conv_0 = tf.keras.layers.Conv2D(self.num_filters, kernel_size=(self.filter_sizes[0], self.input_embedding_size), padding='valid', kernel_initializer='normal', activation='relu')(cat_embedding)
        conv_1 = tf.keras.layers.Conv2D(self.num_filters, kernel_size=(self.filter_sizes[1], self.input_embedding_size), padding='valid', kernel_initializer='normal', activation='relu')(cat_embedding)
        conv_2 = tf.keras.layers.Conv2D(self.num_filters, kernel_size=(self.filter_sizes[2], self.input_embedding_size), padding='valid', kernel_initializer='normal', activation='relu')(cat_embedding)
        conv_3 = tf.keras.layers.Conv2D(self.num_filters, kernel_size=(self.filter_sizes[3], self.input_embedding_size), padding='valid', kernel_initializer='normal', activation='relu')(cat_embedding)

        maxpool_0 = tf.keras.layers.MaxPool2D(pool_size=(self.max_transaction_history*2 - self.filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
        maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(self.max_transaction_history*2 - self.filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
        maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(self.max_transaction_history*2 - self.filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
        maxpool_3 = tf.keras.layers.MaxPool2D(pool_size=(self.max_transaction_history*2 - self.filter_sizes[3] + 1, 1), strides=(1,1), padding='valid')(conv_3)

        concatenated_tensor = tf.keras.layers.concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
        flatten = tf.keras.layers.Flatten()(concatenated_tensor)
        
        return flatten
    
    def create_input(self):
        
        transaction_cols = [x for x in range(self.max_transaction_history)]
        promotion_click_cols = [x for x in range(self.max_promotion_click_history)]
        seq_category_cols = [transaction_cols, promotion_click_cols]
        
        seqs = []
        for i, grp in enumerate(seq_category_cols):
            seqs.append(self.seq_categorical_input('seq_categorical_' + str(i), len(grp)))

        singles = []
        if self.single_categorical_features:
            for col in self.single_categorical_features:
                singles.append(self.singe_categorical_input(str(col), self.single_categorical_features[col][0],
                                                           self.single_categorical_features[col][1]))
        inp_layer =  [s[0] for s in seqs]
        inp_embed = [s[1] for s in seqs]
               
        return inp_layer, inp_embed, singles
    
    
    def seq_categorical_input(self, name, max_history):
    
        seq = tf.keras.layers.Input(shape=(max_history,), dtype='int32', name=name)
        input_embeddings = self.category_embeddings(seq)
        return seq, input_embeddings 

    
    def singe_categorical_input(self, name, unique_size, embedding_size):
        single = tf.keras.layers.Input(shape=(1,), dtype='int32', name=name)
        embeddings = tf.keras.layers.Embedding(output_dim = embedding_size, input_dim = unique_size, 
                           input_length=1, name=name + '_embedding')(single)
        embeddings = tf.keras.layers.Flatten(name = 'flatten_' + name)(embeddings)
        return single, embeddings
    
    def continous_inputs(self, size=None, name='numeric'):
        inp = tf.keras.layers.Input(shape=(size,), dtype='float32', name=name)
        return inp, inp
