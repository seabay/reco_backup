import numpy as np
import keras
from itertools import islice

class DataGenerator(keras.utils.Sequence):
    def __init__(self, path, row_size, batch_size=32, n_classes=10, 
    seq_category_pos=None, categorical_pos=None, numeric_pos=None, shuffle=True):
        'Initialization'
        self.path = path
        self.batch_size = batch_size
        self.row_size = row_size
        self.n_classes = n_classes
        self.seq_category_pos=seq_category_pos
        self.categorical_pos=categorical_pos
        self.numeric_pos=numeric_pos
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.row_size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        pos = (index*self.batch_size, min(self.row_size, (index+1)*self.batch_size))
        X, y = self.__data_generation(pos)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.row_size)  ### bugs here
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, pos):
        
        fts1 = []
        fts2 = []
        fts3 = []
        fts4 = []
        fts5 = []
        fts6 = []
        fts7 = []
        labels = []

        with open(self.path) as f:
            for line in islice(f, pos[0], pos[1]):
                line = line.strip().split(" ")
                label = line[-1]
                
                # update our corresponding batches lists
                fts1.append(np.array([int(float(x)) for x in line[0:self.seq_category_pos[0]]], dtype="uint32"))
                fts2.append(np.array([int(float(x)) for x in line[self.seq_category_pos[0]:self.seq_category_pos[1]]], dtype="uint32"))
                fts3.append(np.array([int(float(x)) for x in line[self.seq_category_pos[1]:self.seq_category_pos[2]]], dtype="uint32"))
                fts4.append(np.array(int(float(line[self.categorical_pos[0]])), dtype="uint32"))
                fts5.append(np.array(int(float(line[self.categorical_pos[1]])), dtype="uint32"))
                fts6.append(np.array(int(float(line[self.categorical_pos[2]])), dtype="uint32"))
                fts7.append(np.array(line[self.numeric_pos[0]:-1]))
            
                labels.append(int(float(label)))
            
            
            # one-hot encode the labels
        labels = keras.utils.to_categorical(labels, num_classes=self.n_classes)
 
        # yield the batch to the calling function
        return ([np.array(fts1), np.array(fts2), np.array(fts3), np.array(fts4), np.array(fts5), 
               np.array(fts6), np.array(fts7)], labels)



import threading

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Py3
        return next(self.it)
        
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def data_generator(inputPath, batch_size, seq_category_pos=None, categorical_pos=None, numeric_pos=None, mode="train"):
    
    f = open(inputPath, "r")
    
    while True:
        
        fts1 = []
        fts2 = []
        fts3 = []
        fts4 = []
        fts5 = []
        fts6 = []
        fts7 = []
        labels = []
        
        # keep looping until we reach our batch size
        while len(labels) < batch_size:
            line = f.readline()
            if line == "":
                f.seek(0)
                line = f.readline()
                if mode == "eval":
                    break
 
            line = line.strip().split(" ")
            label = line[-1]
            
            # update our corresponding batches lists
            fts1.append(np.array([int(float(x)) for x in line[0:seq_category_pos[0]]], dtype="uint32"))
            fts2.append(np.array([int(float(x)) for x in line[seq_category_pos[0]:seq_category_pos[1]]], dtype="uint32"))
            fts3.append(np.array([int(float(x)) for x in line[seq_category_pos[1]:seq_category_pos[2]]], dtype="uint32"))
            fts4.append(np.array(int(float(line[categorical_pos[0]])), dtype="uint32"))
            fts5.append(np.array(int(float(line[categorical_pos[1]])), dtype="uint32"))
            fts6.append(np.array(int(float(line[categorical_pos[2]])), dtype="uint32"))
            fts7.append(np.array(line[numeric_pos[0]:-1]))
            labels.append(int(float(label)))
            
            
            # one-hot encode the labels
        labels = keras.utils.to_categorical(labels, num_classes=100)
 
        # yield the batch to the calling function
        yield ([np.array(fts1), np.array(fts2), np.array(fts3), np.array(fts4), np.array(fts5), 
               np.array(fts6), np.array(fts7)], labels)