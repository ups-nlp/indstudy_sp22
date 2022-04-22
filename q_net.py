from hashlib import new
import os
from os.path import exists
from telnetlib import SE

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Dense, LSTM, AveragePooling1D, Input, BatchNormalization, Dropout, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy 
from tensorflow.keras.optimizers import Adam

##
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
##


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
    stop_words = ["and", "but", "or", "the", "a", "an", "to", "is", "as", "of", "seems", "with"]
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

    return data


# def generate_glove_embeddings(sentences):
#     file = open("GloVe-master/text.txt", 'a')
#     file.writelines(sentences)

#     os.system("cd GloVe-master/")
#     os.system("./demo.sh")
#     os.system("cd ..")

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

    return word2id, w

def get_training_data(data, word_to_id, embedding_matrix):
    lstm_dict = {}

    train_x = []
    ##lstm_train_y = []
    q_vals = []

    for datum in data:
        cur_state = datum[0]
        rand_act = datum[2]

        if len(cur_state.split(' ')) > 100:
            print(cur_state)

        combined_str = cur_state + ' ' + rand_act

        q_score = datum[3]

        if not (combined_str in lstm_dict):
            ##print("--Generating vector for state:", combined_str)
            result = sentence_to_vect_sequence(combined_str, word_to_id, embedding_matrix) #sentence_to_vect(combined_str, word_to_id, embedding_matrix, lstm)
            if result != []:
                lstm_dict[combined_str] = result
                ##print("---Generated")     
                ##print(result)
            ##else:
                ##print(result)
                ##print("---Failed to generate")
        ##else:
            ##print("--Found vector for state:", combined_str)

        # if not (rand_act in lstm_dict):
        #     ##print("--Generating vector for state:", combined_str)
        #     print(rand_act)
        #     res = sentence_to_vect_sequence(rand_act, word_to_id, embedding_matrix)
        #     if len(res) != 0:
        #         lstm_dict[rand_act] = res[0]
    
        if combined_str in lstm_dict: ##and rand_act in lstm_dict:

           #print(lstm_dict[combined_str].shape)

            train_x.append(lstm_dict[combined_str])
            ##lstm_train_y.append(lstm_dict[rand_act])
            q_vals.append(q_score)

            # if len(lstm_dict.keys()) == 10:
            #     break

    ##print("--------------------------", len(train_x), len(q_vals))
    return (train_x, q_vals), lstm_dict ##(lstm_train_x, lstm_train_y), lstm_dict, q_vals
    
## MAJORLY NEEDS LOOKED AT 
## ISSUE IS THAT I NEED TO PASS LSTM SEQUENCE OF WORD VECTORS (sentence) TO OUTPUT SENTENCE STATE VECTOR
# NEEDS TO FIGURE OUT HOW TO GET OUTPUT TO FIT THOUGH WHAT THE HELL
def sentence_to_vect_sequence(sentence, word_to_id, embedding_matrix):  
    ## sentence_vect = np.empty([1, 50, 1])
    word_ls = sentence.split(' ')
    word_vect_sequence = []
    for word in word_ls:
        if not word in word_to_id.keys(): # if we do not know a word then ignore it ## best solution?
            # print("---Vector for word not found:", word)
            # return np.array([])
            continue

        ##print(embedding_matrix[word_to_id[word]])

        word_vect_sequence.append(embedding_matrix[word_to_id[word]])

    ##print(word_vect_sequence)

    return word_vect_sequence


def save_dict(file_name, dict_to_write):
    f = open(file_name, 'a')
    
    for key in dict_to_write.keys():
        f.write(key + "~" + np.array2string(dict_to_write[key]) + '\n')

    f.close()

def load_dict(file_name):
    new_dict = {}
    
    f = open(file_name, 'r')

    c = 0

    for line in f:
        ##print('-', c)
        
        line_split = line.replace("\n", '').split('~')
        ##print(line_split)
        new_dict[line_split[0]] = np.fromstring(line_split[1])

        c += 1

    return new_dict

def clean_sent_to_input(sentence, lstm_dict):
    if sentence in lstm_dict.keys():
        return lstm_dict.keys()
    
    return None

def pad_lstm_input(x_train):
    max_len = 0

    for x in x_train:
        if len(x) > max_len:
            max_len = len(x)

    for i in range(len(x_train)):
        while(len(x_train[i]) < max_len):
            x_train[i].append([0] * 50) 

    return x_train, max_len

def generate_net():
    # Hyper Params (need adjust)
    layer_one_nodes = 32
    layer_two_nodes = 16
    learning_rate = 0.005
    ##output_size = 1

    # print("Making Q-Net")
    model = Sequential()

    model.add(Masking(mask_value = 0.0))

    model.add(LSTM(10, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(10, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(AveragePooling1D())

    model.add(layers.Dense(layer_one_nodes, activation = 'relu'))
    ## model.add(BatchNormalization())
    model.add(layers.Dense(layer_two_nodes, activation = 'relu'))
    ## model.add(BatchNormalization())

    model.add(layers.Dense(1)) #, activation = 'sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate = learning_rate),
        metrics=['accuracy'],   
    )

    return model

# def generate_lstm(training): ### FOR LSTM train on sentence vect label is the action

#     x_train, y_train = training

#     input_size = len(x_train)

#     ##print(type(x_train))
#     ##print(type(y_train))

#     ##print(x_train[0])

#     x_train, batch_size = pad_lstm_input(x_train)
    
#     ##print(x_train[0])

#     ##print('-------', input_size)

    
#     x_train = np.array(padded_x_train)
#     y_train = np.array(y_train)

#     print("--------------\n",x_train.shape,"\n--------------")
#     print("--------------\n",y_train.shape,"\n--------------")

#     ##print(x_train.shape)

#     #x_train.reshape(input_size, batch_size, 50)

#     learning_rate = 0.005
#     epochs = 20

#     lstm = Sequential()

#     lstm.add(LSTM(10, input_shape=(batch_size, 50), return_sequences=True))
#     ## lstm.add(BatchNormalization())
#     lstm.add(AveragePooling1D())

#     # More layers + dropout? 20%?
    
#     lstm.add(Dense(1))

#     lstm.compile(
#         loss='binary_crossentropy',
#         optimizer=Adam(learning_rate = learning_rate)  
#     )

#     # NEEDS FIT
#     lstm.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)

#     ##print("--------------\n",y_train.shape,"\n--------------")

#     return lstm, y_train


def sentence_through_lstm(sequence_vect, lstm, batch_size):

    reformated_input = np.array([sequence_vect]).reshape(1, batch_size, 50)
    ##print("++++++++++++++++", reformated_input.shape)
    st = lstm.predict(reformated_input)
    print(st.shape)
    val = st[0].reshape(50)

    ##print(val)

    return val


def compile_corupus():
    dir_name = "data"
    files = os.listdir(dir_name)

    write_str = ""
    punctuation_to_remove = ['.', ':', '?', ',', ';', '"', "'", '|', '*', "'s"]

    for file in files:
        f = open(dir_name + '/' + file, 'r')

        for line in f:
            l = ' '.join(line.lower().split('$')[:3])

            for punct in punctuation_to_remove:
                l = l.replace(punct, '')

            write_str += l

    dest = open("GloVe-master/text.txt", 'w')
    dest.write(write_str)


##################################################
data = read_data("data")

if not exists("GloVe-Master/text.txt"):
    # generate_glove_embeddings(sentences)
    # NEED LOWERCASE NO PUNCT FROM ALL DATAFILES INTO text.txt in glove dir
    compile_corupus()
    print("needs GLOVE Embeddings")
    
else:
    print("-Compiling Embeddings")
    word_to_id, embedding_matrix = compile_embeddings()

    print("-Getting Training Data")
    lstm_training, lstm_dict = get_training_data(data, word_to_id, embedding_matrix) ##, q_values = get_training_data(data, word_to_id, embedding_matrix)

    padded_state_vects, max_sen_len = pad_lstm_input(lstm_training[0])
    print("---------------------------MAX LEN", max_sen_len)

    state_vects = np.array(padded_state_vects, 'float32') ##.astype("float32")
    q_values = np.array(lstm_training[1], 'float32') ##.astype("float32")

    # for i in state_vects:
    #     print(len(i))

    ##print('-=-=-=-=-=-=-=-=-=', type(state_vects), type(q_values))

    # print("-Making LSTM")
    # lstm, action_vects = generate_lstm(lstm_training)
    print("-Generating Net")
    model = generate_net()

    print("-Training Net")
    epochs = 20
    batch_size = 30
    ##print(state_vects.shape, q_values.shape)
    print(state_vects.shape)
    model.fit(state_vects, q_values, epochs = epochs) #, batch_size = batch_size)

    print(model.summary())

    print("-All Done... saving")
    net_dir = "nets"
    cur_file_path = "nets/"
    net_file_name_format = "lstm"
    file_num = 0
    while(exists(cur_file_path)):
        file_num += 1
        cur_file_path = net_dir + '/' + net_file_name_format + str(file_num)

    model.save(cur_file_path)


    ##print("-Loading lstm model")
    ##lstm = keras.models.load_model("nets/lstm1")


    #print("-Saving lstm dict")
    #save_dict("data/-lstm_dict.txt", lstm_dict) # needs fix

    # x_train, batch_size = pad_lstm_input(lstm_training[0])
    # input = [sentence_through_lstm(x, lstm, batch_size) for x in lstm_training[0]]
    # print(input[0].dtype)
    # print(input[0])
    # print('------------', len(input))
    # input = np.array(input).reshape(len(input), 50)

    # print(input.shape)
    
    

    #loaded_dict = load_dict("data/-lstm_dict.txt")

    # Hyper Params (need adjust)
    # epochs = 20
    # batch_size = 32
    # layer_one_nodes = 32
    # layer_two_nodes = 16
    # learning_rate = 0.005
    # input_size = 50 #NOTE PLACEHOLDER?
    # output_size = 1

    # # print("Making Q-Net")
    # q_net = Sequential()
    # q_net.add(keras.Input(shape=(50)))
    # q_net.add(layers.Dense(layer_one_nodes, activation = 'relu'))
    # q_net.add(layers.Dense(layer_two_nodes, activation = 'relu'))
    # q_net.add(layers.Dense(output_size, activation = 'sigmoid'))
    # q_net.compile(
    #     loss='binary_crossentropy',
    #     optimizer=Adam(learning_rate = learning_rate),
    #     metrics=['accuracy'],   
    # )
    
    # print("Starting Training")
    # Train model

    # q_values = np.array(q_values).astype('float32')
    # print(q_values.dtype)
    # print(q_values[0])

    # q_net.fit(input, q_values, batch_size = batch_size, epochs = epochs)

    # print("All Done... saving")
    # net_dir = "nets"
    # cur_file_path = "nets/"
    # net_file_name_format = "qNet"
    # file_num = 0
    # while(exists(cur_file_path)):
    #         file_num += 1
    #         cur_file_path = net_dir + '/' + net_file_name_format + str(file_num)

    # q_net.save(cur_file_path)


    # model = keras.models.load_model("nets/qNet1")

    # state = "behind house you are behind white house path leads into forest east in one corner of house there small window which open west"

    # state_vect = np.array()

    # predict = model.predict(state_vect)

    # print(predict)
    
