import os
from os.path import exists
from telnetlib import SE

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

#
def clean_text(text):
    # Lowercase text
    text = text.lower()
    
    new_text = ""

    # Remove punctuation
    punctuation = ['.', '?', '!', '*', '&', '(', ')', ',', ':']
            
    for punct in punctuation:
        text = text.replace(punct, '')

    text = text.replace('|', ' ')

    # Remove stop words
    stop_words = ["and", "but", "or", "the", "a", "an", "to", "is"]
    for word in text.split(' '):
        word = word.strip()
        if (not word in stop_words) and (word != ''):
            new_text += word + ' '

    return new_text.strip()

#
def read_data(dir_name):
    files = os.listdir(dir_name)

    data = []
    sentences = []

    for f in files:
        file = open(dir_name + "/" + f, 'r')
        if f[0] == '-':
            continue

        for line in file:
            cur_obs, next_obs, rand_act, q_score, cur_score, state_status = line.split('$')
            cur_obs = clean_text(cur_obs)
            next_obs = clean_text(next_obs)
            rand_act = clean_text(rand_act)

            data.append((cur_obs, next_obs, rand_act, q_score, cur_score, state_status))
            
            sentences.append(cur_obs)
            sentences.append(" " + rand_act + " ")

    file.close()

    return (data, sentences)


def generate_glove_embeddings(sentences):
    file = open("GloVe-master/text.txt", 'a')
    file.writelines(sentences)

    os.system("cd GloVe-master/")
    os.system("./demo.sh")
    os.system("cd ..")

# FROM AL CHAMBERS HW 3 BASICALLY
def compile_embeddings():
    with open("GloVe-master/vocab.txt", 'r') as f:
        words = [word.rstrip().split(' ')[0] for word in f.readlines()]

    with open("GloVe-master/vectors.txt", 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            word = vals[0]
            vec = vals[1:]
            vectors[word] = [float(x) for x in vec]

    vocabSize = len(words)
    word2id = {w: idx for idx, w in enumerate(words)}
    id2word = {idx: w for idx, w in enumerate(words)}
    vectorDim = len(vectors[id2word[0]])

    w = np.zeros((vocabSize, vectorDim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        w[word2id[word], :] = v
        
    wNorm = np.zeros(w.shape) # needed?
    d = (np.sum(w **2, 1) ** (0.5)) # needed?
    wNorm = (w.T / d).T # needed?

    return (word2id, w)

# Generate network
# def train_q_network():
#     # Hyper params
#     batch_size = 64
#     learning_rate = 0.001
#     num_episodes = 10

#     fc_layer_params = (100, 50)

#     dense_layers = [make_dense_layer(num_units) for num_units in fc_layer_params]
#     q_values_layer = tf.keras.layers.Dense(
#         num_actions=2, #change perhaps ????
#         activation = None,
#         kernel_initializer = tf.keras.initializers.RandomUniform(
#             minval = -0.03, maxval = 0.03
#         ),
#         bias_initializer = tf.keras.initializers.Constant(-0.2)
        
#     )
#     q_net = Sequential(dense_layers + [q_values_layer])

#     optmizer = Adam(learning_rate = learning_rate)
#     train_step_counter = tf.Variable(0)

#     agent = q_net.DqnAgent 

def make_dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distributions='truncated_normal'
        )
    )
    
def get_training_data(data, word_to_id, embedding_matrix, lstm):
    lstm_dict = {}

    inputs = np.empty([1])
    q_vals = np.array([1])

    for datum in data:
        cur_state = datum[0]
        rand_act = datum[2]

        combined_str = cur_state + " " + rand_act

        q_score = datum[3]

        if not (combined_str in lstm_dict):
            print("--Generating vector for state:", combined_str)
            result = sentence_to_vect(combined_str, word_to_id, embedding_matrix, lstm)
            if not np.array_equal(result, np.array([])):
                lstm_dict[combined_str] = result
                print("---Generated")     
            else:
                print(result)
                print("---Failed to generate")
        else:
            print("--Found vector for state:", combined_str)
        
        if combined_str in lstm_dict.keys():
            np.append(inputs, lstm_dict[combined_str])
            np.append(q_vals, q_score)

    return (inputs, q_vals)
    

def sentence_to_vect(sentence, word_to_id, embedding_matrix, lstm):
    sentence_vect = np.empty([1, 50, 1])
    word_ls = sentence.split(' ')
    for word in word_ls:
        if not word in word_to_id.keys():
            print("---Vector for word not found:", word)
            return np.array([])

        np.add(sentence_vect, lstm(embedding_matrix[word_to_id[word]].reshape(1, 50, 1)))

    #print(sentence_vect)

    for i in range(0, 50):
        sentence_vect[0][i][0] = sentence_vect[0][i][0] / len(word_ls)

    return sentence_vect[0]


##################################################
data, sentences = read_data("data")



if not exists("GloVe-Master/text.txt"):
    generate_glove_embeddings(sentences)
    
else:
    word_to_id, embedding_matrix = compile_embeddings()

    #print(word_to_id.keys())

    # LSTM
    
    #index = 0
    #embeddings = embedding_matrix[index].reshape(1, 50, 1)

    print("-Making LSTM")
    lstm = Sequential()
    lstm.add(LSTM(10, input_shape=(50, 1)))
    lstm.add(Dense(1))
    
    #output = lstm(embeddings)
    #print(output)

    print("-Getting Training Data")
    inputs, labels = get_training_data(data, word_to_id, embedding_matrix, lstm)


    # Hyper Params (need adjust)
    epochs = 20
    batch_size = 32
    layer_one_nodes = 32
    layer_two_nodes = 16
    learning_rate = 0.005
    input_size = 50 #NOTE PLACEHOLDER?
    output_size = 1

    print(inputs.shape)

    print("Making Q-Net")
    q_net = Sequential()
    q_net.add(keras.Input(shape=(50, 1)))
    q_net.add(layers.Dense(layer_one_nodes, activation = 'relu'))
    q_net.add(layers.Dense(layer_two_nodes, activation = 'relu'))
    q_net.add(layers.Dense(output_size, activation = 'sigmoid'))

    # obs_input = keras.layers.Input(shape=(1,))
    # q_input = keras.layers.Input(shape=(1,))
    # dual_input = keras.layers.Concatenate(axis=1)([obs_input, q_input])
    # dense_1 = layers.Dense(layer_one_nodes, activation = 'relu', input_dim = 2) (dual_input)
    # dense_2 = layers.Dense(layer_one_nodes, activation = 'relu', input_dim = 2) (dense_1)
    # output = layers.Dense(output_size, activation = 'sigmoid', input_dim = 2) (dense_2)
    # q_net = keras.models.Model(inputs=[obs_input, q_input], output = output)

    q_net.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate = learning_rate),
        metrics=['accuracy'],   
    )

    
    # Train model
    training = np.array(inputs)


    print("Starting Training")

    q_net.fit(training, labels, batch_size = batch_size, epochs = epochs)

    print("All Done... saving")

    q_net.save("nets")
    
