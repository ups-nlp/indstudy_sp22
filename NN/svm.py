"""
Created 22.2.21

@author Eric Markewitz
"""

import numpy as np
from operator import add
from operator import truediv
import re

#import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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
        else:
            print("Word not in the vocab: " + word)

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
    module = re.sub('\n', '', module)

    trainingInputs.append(obs_vect)
    trainingOutputs.append(module)

    """
    print("obs vect:  ")
    print(obs_vect)
    print()
    print("module " + module)
    print()
    print()
    print()

    i+=1
    if i > 5:
        break
    """
"""
print()
print(trainingInputs)
print(trainingOutputs)

"""
np_input = np.array(trainingInputs)
np_output = np.array(trainingOutputs)


train_obs, test_obs, train_labels, test_labels = train_test_split(np_input, np_output, test_size = 0.2)

model = SVC()

model.fit(train_obs, train_labels)

predictions = model.predict(test_obs)

print(classification_report(test_labels,predictions))




i=0
while i<40:
    #print(type(test_obs[i]))
    if(test_labels[i] != predictions[i]):
        print("Test observation: " + vect_to_sentince[str(test_obs[i].tolist())])
        print("Test label " + str(test_labels[i]))
        print("Prediction: " + str(predictions[i]))
    i+=1



"""
import matplotlib
import matplotlib.pyplot as plt
%matplotlib notebook

from sklearn.decomposition import PCA

tsne = TSNE(n_components=2)
W_proj = tsne.fit_transform(W_norm)


i=0
with
for line in vocab

for character in characterNames:
    wid = word2id.get(character)
    coords = W_proj[wid]
    x = coords[0]
    y = coords[1]
    plt.scatter(x, y, marker='x', color='red')
    plt.text(x+0.003, y+0.003, character, fontsize=9)
"""
