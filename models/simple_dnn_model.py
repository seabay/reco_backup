import numpy as np
import builtins
import tensorflow as tf

class RecoDNN():
    
    def __init__(self, max_transaction_history = 50, max_product_click_history = 50, max_promotion_click_history = 50,
                 category_size = 100, single_categorical_features = None, numeric_features_size = 10,
                 hidden_layer1_size = 256, hidden_layer2_size = 128, hidden_layer3_size = 64, activation='relu',
                input_embedding_size = 128, multi_gpu_model=False):
        
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
        
        self.category_embeddings = tf.keras.layers.Embedding(output_dim=self.input_embedding_size, input_dim = self.category_size, 
                       input_length = builtins.max(self.max_transaction_history, self.max_product_click_history, self.max_promotion_click_history), mask_zero=True, name='category_embeddings')
        
        self.model = None
        self.build()
        
    
    def build(self):
        
        inp_layer, inp_embed = self.create_input()
        
        v = tf.keras.layers.Dense(self.hidden_layer1_size, activation = self.activation)(tf.keras.layers.concatenate(inp_embed)) 
        v = tf.keras.layers.Dense(self.hidden_layer2_size, activation = self.activation)(v)
        v = tf.keras.layers.Dense(self.hidden_layer3_size, activation = self.activation, name='user_embedding')(v)
        output = tf.keras.layers.Dense(self.category_size, activation ='softmax', name='softmax_layer')(v)
        self.model = tf.keras.models.Model(inputs = inp_layer, outputs = [output])   

        if self.multi_gpu_model:
            try:
                self.model = multi_gpu_model(self.model, gpus=2, cpu_relocation=True)
                print("Training using multiple GPUs..")
            except:
                print("Training using single GPU or CPU..")
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    
    def create_input(self):
        
        transaction_cols = [x for x in range(self.max_transaction_history)]
        product_click_cols = [x for x in range(self.max_product_click_history)]
        promotion_click_cols = [x for x in range(self.max_promotion_click_history)]
        seq_category_cols = [transaction_cols, product_click_cols, promotion_click_cols]
        
        seqs = []
        for i, grp in enumerate(seq_category_cols):
            seqs.append(self.seq_categorical_input('seq_categorical_' + str(i), len(grp)))

        singles = []
        if self.single_categorical_features:
            for col in self.single_categorical_features:
                singles.append(self.singe_categorical_input(str(col), self.single_categorical_features[col]))

        nums = self.continous_inputs(self.numeric_features_size)

        inp_layer =  [s[0] for s in seqs]
        inp_layer += [s[0] for s in singles]
        inp_layer.append(nums[0])
        inp_embed = [s[1] for s in seqs]
        inp_embed += [s[1] for s in singles]
        inp_embed.append(nums[1])
               
        return inp_layer, inp_embed
    
    
    def seq_categorical_input(self, name, max_history):
    
        seq = tf.keras.layers.Input(shape=(max_history,), dtype='int32', name=name)
        input_embeddings = self.category_embeddings(seq)
        avg_embedding = tf.keras.layers.GlobalAveragePooling1D(name=name + '_avg_embedding')(input_embeddings, mask=self.category_embeddings.compute_mask(seq))
        #max_embedding = tf.keras.layers.GlobalMaxPooling1D(name=name + '_max_embedding')(input_embeddings)

        return seq, avg_embedding   #keras.layers.add([avg_embedding, max_embedding])

    
    def singe_categorical_input(self, name, unique_size):
        single = tf.keras.layers.Input(shape=(1,), dtype='int32', name=name)
        embeddings = tf.keras.layers.Embedding(output_dim = self.input_embedding_size, input_dim = unique_size, 
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
        self.model.save_weights(checkpoint_path)
        print("Model saved")


    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")
