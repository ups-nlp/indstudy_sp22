"""
Created 22.2.25

@author Eric Markewitz
"""

import numpy as np
from operator import add
from operator import truediv
import re

#import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt


trainingInputs = []
trainingOutputs = []

dm_training_data = open("../data/walkthrough_training_data.txt")

def embed_vocab() -> (list,dict):
    """
    Reads in the vocab and vector GloVemaster files in from the data folder.
    Returns a dictionary matching a word to an index and a list of the vectors

    @return list: A normalized list of vectors, one for each word
    @return dict: A dictionary with a word as a key and the id for the list as the value
    """
    with open("../data/vocab.txt", 'r') as f:
        #Read in the list of words
        words = [word.rstrip().split(' ')[0] for word in f.readlines()]

    with open("../data/vectors.txt", 'r') as f:
        #word --> [vector]
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            word = vals[0]
            vec = vals[1:]
            vectors[word] = [float(x) for x in vec]

    #Compute size of vocabulary
    vocab_size = len(words)
    word2id = {w: idx for idx, w in enumerate(words)}
    id2word = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[id2word[0]])
    #print("Vocab size: " + str(vocab_size))
    #print("Vector dimension: " + str(vector_dim))

    #Create a numpy matrix to hold word vectors
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[word2id[word], :] = v

    #Normalize each word vector to unit length
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T

    return W_norm, word2id

def create_vect(vocab_vectors:list, word2id:dict, observation:str, sentince_to_vect:dict, vect_to_sentince:dict) -> list:
    """
    Takes an observation and returns a 50 dimensional vector representation of it

    @param str: a string containing an observation

    @return list: A list representing a 50 dimensional normalized vector
    """
    obs_split = observation.split(' ')
    num_words = 0

    #Creates an empty list of size 50 to be filled in
    vect_size = 50
    avg_vect = [0] * vect_size

    for word in obs_split:
        #Check if the word is in our vocab, if it is add it to the vector
        if(word2id.get(word) is not None):
            id = word2id.get(word)
            norm_vect = vocab_vectors[id]

            avg_vect = list(map(add, avg_vect, norm_vect))
            num_words +=1
        #else:
            #print("Word not in the vocab: " + word)

    words = [num_words] * vect_size
    avg_vect = list(map(truediv, avg_vect, words))

    sentince_to_vect[observation] = avg_vect
    vect_to_sentince[str(avg_vect)] = observation

    return(avg_vect)


#Start of script
vocab_vectors, word2id = embed_vocab()

sentince_to_vect = {}
vect_to_sentince = {}


i = 0
for line in dm_training_data:
    lst = line.split(',')
    observation = lst[0]
    obs_vect = create_vect(vocab_vectors, word2id, observation, sentince_to_vect, vect_to_sentince)
    action = lst[1]
    module = lst[2]
    module_num = int(module)

    trainingInputs.append(obs_vect)
    trainingOutputs.append(module_num)


np_input = np.array(trainingInputs)
np_output = np.array(trainingOutputs)

train_obs, test_obs, train_labels, test_labels = train_test_split(np_input, np_output, test_size = 0.2)

#Hyperparameters
epochs = 20
batch_size = 32
hidden_lyr1_nodes = 32
hidden_lyr2_nodes = 32
learning_rate = 0.0005
input_size = 50
output_size = 1

"""
print(train_labels.shape)
t_cat = to_categorical(train_labels)
print(t_cat.shape)
print(t_cat[0])
print(t_cat[1])
print(t_cat[2])
print(t_cat[3])
"""

def create_model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_size,)))
    model.add(layers.Dense(hidden_lyr1_nodes, activation='relu'))
    model.add(layers.Dense(hidden_lyr2_nodes, activation='relu'))
    model.add(layers.Dense(output_size, activation='sigmoid'))

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    #evaluation = model.evaluate(x=test_obs,y=test_labels, verbose=1)

    return model



"""
# Plot the accuracy of the model as it trains
plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
"""

keras_reg = KerasClassifier(build_fn= create_model, epochs=epochs, batch_size=batch_size, verbose=0)

adaBoost = AdaBoostClassifier(base_estimator=keras_reg)

adaBoost.fit(train_obs, train_labels)

predictions = adaBoost.predict(test_obs)

print(classification_report(test_labels,predictions))
