import numpy as np
import builtins
import tensorflow as tf

import models.losses as losses

class RecoDNN():
    
    def __init__(self, max_transaction_history = 20, max_product_click_history = 20, max_promotion_click_history = 20,
                 category_size = 100, single_categorical_features = None, numeric_features_size = 1,
                 hidden_layer1_size = 256, hidden_layer2_size = 128, hidden_layer3_size = 64, activation='relu',
                input_embedding_size = 64, seq_pooling_mode='cat', multi_gpu_model=False):
        
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
        self.seq_pooling_mode = seq_pooling_mode
        self.multi_gpu_model = multi_gpu_model
        
        self.category_embeddings = tf.keras.layers.Embedding(input_dim = self.category_size, output_dim=self.input_embedding_size,  input_length=20, mask_zero=True, name='category_embeddings')
        
        self.model = None

        self.build()
        
    
    def build(self):
        
        inp_layer, inp_embed = self.create_input()
        
        v = tf.keras.layers.Dense(512, activation = self.activation)(tf.keras.layers.concatenate(inp_embed)) 
        v = tf.keras.layers.LayerNormalization()(v)
        v = tf.keras.layers.Dense(self.hidden_layer1_size, activation = self.activation)(v)
        v = tf.keras.layers.LayerNormalization()(v)
        v = tf.keras.layers.Dense(self.hidden_layer2_size, activation = self.activation)(v)
        v = tf.keras.layers.LayerNormalization()(v)
        v = tf.keras.layers.Dense(self.hidden_layer3_size, activation = self.activation, name='user_embedding')(v)
        v = tf.keras.layers.LayerNormalization()(v)
        output = tf.keras.layers.Dense(self.category_size, activation ='softmax', name='softmax_layer')(v)
        self.model = tf.keras.models.Model(inputs = inp_layer, outputs = [output])   

        if self.multi_gpu_model:
            
            try:
                self.model = tf.keras.utils.multi_gpu_model(self.model, gpus=8, cpu_relocation=True)
                print("Training using multiple GPUs..")
            except:
                print("Training using single GPU or CPU..")
            
            #self.model = tf.keras.utils.multi_gpu_model(self.model, gpus=8, cpu_relocation=True)

        #self.model.compile(loss=[losses.categorical_focal_loss(1，1)], optimizer=tf.keras.optimizers.Adam(lr=0.005), metrics=['accuracy'])
        self.model.compile(loss=['sparse_categorical_crossentropy'], optimizer=tf.keras.optimizers.Adam(lr=0.005), metrics=['accuracy'])

    
    def create_input(self):
        
        transaction_cols = [x for x in range(self.max_transaction_history)]
        promotion_click_cols = [x for x in range(self.max_promotion_click_history)]
        seq_category_cols = [transaction_cols, promotion_click_cols]
        
        seqs = []
        for i, grp in enumerate(seq_category_cols):
            seqs.append(self.seq_categorical_input('seq_categorical_' + str(i), len(grp), self.seq_pooling_mode))

        singles = []
        if self.single_categorical_features:
            for col in self.single_categorical_features:
                singles.append(self.singe_categorical_input(str(col), self.single_categorical_features[col][0], self.single_categorical_features[col][1]))
        inp_layer =  [s[0] for s in seqs]
        inp_embed = [s[1] for s in seqs]
               
        return inp_layer, inp_embed
    
    
    def avg_pooling(self, name, max_history):
        seq = tf.keras.layers.Input(shape=(max_history,), dtype='int32', name=name)
        category_embeddings = tf.keras.layers.Embedding(input_dim = self.category_size, output_dim=self.input_embedding_size, mask_zero=True,name=name+'category_embeddings')
        input_embeddings = category_embeddings(seq)
        avg_embedding = tf.keras.layers.GlobalAveragePooling1D(name=name + '_avg_embedding')(input_embeddings, mask=self.category_embeddings.compute_mask(seq))
        return seq, avg_embedding

    def max_pooling(self, name, max_history):
        seq = tf.keras.layers.Input(shape=(max_history,), dtype='int32', name=name)
        category_embeddings = tf.keras.layers.Embedding(input_dim = self.category_size, output_dim=self.input_embedding_size, name=name+'category_embeddings')
        input_embeddings = category_embeddings(seq)
        max_embedding = tf.keras.layers.GlobalMaxPooling1D(name=name + '_max_embedding')(input_embeddings)
        #maxf = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=1), name = name + '_max_embedding')
        #max_embedding = maxf(input_embeddings)
        return seq, max_embedding

    def cat_pooling(self, name, max_history):
        seq = tf.keras.layers.Input(shape=(max_history,), dtype='int32', name=name)
        category_embeddings = tf.keras.layers.Embedding(input_dim = self.category_size, output_dim=self.input_embedding_size, name=name+'category_embeddings')
        input_embeddings = category_embeddings(seq)
        return seq, tf.keras.layers.Flatten(name = 'flatten_' + name)(input_embeddings)

    def seq_categorical_input(self, name, max_history, mode='avg'):

        if mode == 'avg':
            return self.avg_pooling(name, max_history)
        elif mode == 'max':
            return self.max_pooling(name, max_history)
        elif mode == 'cat':
            return self.cat_pooling(name, max_history)
        else:
            raise Exception('Not support ' + mode)
    
    def singe_categorical_input(self, name, unique_size, embedding_size):
        single = tf.keras.layers.Input(shape=(1,), dtype='int32', name=name)
        embeddings = tf.keras.layers.Embedding(output_dim = embedding_size, input_dim = unique_size, 
                           input_length=1, name=name + '_embedding')(single)
        embeddings = tf.keras.layers.Flatten(name = 'flatten_' + name)(embeddings)
        return single, embeddings
    
    def continous_inputs(self, size=None, name='numeric'):
        inp = tf.keras.layers.Input(shape=(size,), dtype='float32', name=name)
        return inp, inp


    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save(checkpoint_path)
        print("Model saved")


    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model = tf.keras.models.load_model(checkpoint_path)
        print("Model loaded")
