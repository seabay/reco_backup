import tensorflow as tf

class RecoEstimator():
    
    def __init__(self, max_transaction_history = 20, max_product_click_history = 20, max_promotion_click_history = 20,
                 category_size = 100, single_categorical_features = None, numeric_features_size = 10,
                 hidden_layer1_size = 1024, hidden_layer2_size = 512, hidden_layer3_size = 256, activation='relu',
                input_embedding_size = 128):
        
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
        self.model = None
        self.build()
        
    
    def build(self):
        seqs = self.create_input() 
        self.model = tf.estimator.DNNClassifier(feature_columns=seqs,n_classes=self.category_size,
                                               hidden_units=[self.hidden_layer1_size, self.hidden_layer2_size, self.hidden_layer3_size],
                                               optimizer=tf.compat.v1.train.AdamOptimizer(0.0005))
    def create_input(self):
        seqs=[]
        seqs.append(self.seq_categorical_input('txSeq'))
        seqs.append(self.seq_categorical_input('clickSeq'))
        
        if self.single_categorical_features:
            for col in self.single_categorical_features:
                seqs.append(self.single_categorical_input(str(col), self.single_categorical_features[col][0], self.single_categorical_features[col][1]))

        return seqs
        
    def seq_categorical_input(self, name):
        seq_input = tf.feature_column.categorical_column_with_identity(name, self.category_size)
        seq_emb = tf.feature_column.embedding_column(categorical_column=seq_input, dimension=self.input_embedding_size)
        return seq_emb
    
    def single_categorical_input(self, name, value_size, embedding_size):
        single_input = tf.feature_column.categorical_column_with_identity(name, value_size)
        single_emb = tf.feature_column.embedding_column(categorical_column=single_input, dimension=embedding_size)
        return single_emb