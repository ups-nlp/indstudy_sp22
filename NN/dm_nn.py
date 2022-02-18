"""
Original created 21.12.3
Recreated 22.2.6

@author: Eric Markewitz
"""

import numpy as np
from operator import add
from operator import truediv
import re

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


#Decisionmaker NEURAL NET
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

def create_vect(vocab_vectors:list, word2id:dict, observation:str) -> list:
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
        else:
            print("Word not in the vocab: " + word)

    words = [num_words] * vect_size
    avg_vect = list(map(truediv, avg_vect, words))

    return(avg_vect)


#Start of script
vocab_vectors, word2id = embed_vocab()

for line in dm_training_data:
    lst = line.split(',')
    observation = lst[0]
    obs_vect = create_vect(vocab_vectors, word2id, observation)
    action = lst[1]
    module = lst[2]
    module = re.sub('\n', '', module)

    trainingInputs.append(obs_vect)
    trainingOutputs.append(module)


np_input = np.array(trainingInputs)
np_output = np.array(trainingOutputs)

train_obs, test_obs, train_labels, test_labels = train_test_split(np_input, np_output, test_size = 0.2, random_state =1)

#Hyperparameters
epochs = 20
batch_size = 32
hidden_lyr1_nodes = 32
hidden_lyr2_nodes = 16
learning_rate = 0.005
input_size = 50
output_size = 2

# Build the model
model = Sequential([
  Dense(hidden_lyr1_nodes, activation='relu', input_shape=(input_size,)),
  Dense(hidden_lyr2_nodes, activation='relu', input_shape=(hidden_lyr1_nodes,)),
  Dense(output_size, activation='softmax'),
])


# Compile the model.
model.compile(
  optimizer=Adam(lr=learning_rate),
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model. The use of to_categorical converts the indices to the
# actions to one-hot vectors
history = model.fit(train_obs,
                    to_categorical(train_labels),
                    verbose = 1,
                    validation_split = 0.2, # split data in 80/20 sets
                    epochs=epochs,
                    batch_size=batch_size)


# Plot the accuracy of the model as it trains
plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')

# Evaluate the model.
# output: accuracy is 0.425 on test set
model.evaluate(test_obs, to_categorical(test_labels))

model.save('./dm_nn_v2')
