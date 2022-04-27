from cmath import inf
from hashlib import new
import os
from os.path import exists
from telnetlib import SE

import numpy as np

import tensorflow as tf
from tensorflow import keras
from torch import q_scale
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Dense, LSTM, AveragePooling1D, Input, BatchNormalization, Dropout, Masking, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy, Huber
from tensorflow.keras.optimizers import Adam, RMSprop

## IDK WHAT THIS IS LOL VVVV
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
## IDK WHAT THIS IS LOL ^^^^


#
def clean_text(text):
    # Lowercase text
    text = text.lower()
    
    new_text = ""

    # Remove punctuation
    punctuation = ['.', '?', '!', '*', '&', '(', ')', ',', ':']
            
    for punct in punctuation:
        text = text.replace(punct, '')

    text_split = text.split('|')
    text_split_without_long_unecessary_desc = text_split[0:1] + text_split[2:]
    ##print(text_split_without_long_unecessary_desc)
    text = ' '.join(text_split_without_long_unecessary_desc)
    ##print("--", text)

    ##text = text.replace('|', ' ')

    # Remove stop words
    stop_words = ["and", "but", "or", "the", "a", "an", "to", "is", "as", "of", "seems", "with", "in", "it", "there", "on"]
    for word in text.split(' '):
        word = word.strip()
        if (not word in stop_words) and (word != ''):
            new_text += word + ' '

    ##print("---", new_text)
    return new_text.strip()

#
def read_data(dir_name):
    files = os.listdir(dir_name)

    largest_magnitude_q = 0

    ##max_sen_len = 0
    ##avg_sen_len = 0
    ##total_sents = 0

    ###

    max_obs_len = 0
    max_act_len = 0

    data = []

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

            if abs(float(q_score)) > largest_magnitude_q:
                largest_magnitude_q = abs(float(q_score))

            ##full_data_sen = cur_obs + " " + rand_act

            ##sen_len = len(clean_text(full_data_sen).split(' '))

            ##avg_sen_len += sen_len
            ##total_sents += 1
            
            ##if sen_len > max_sen_len: #len(clean_text(full_data_sen)) > max_sen_len:
                ##max_sen_len = sen_len ##len(clean_text(full_data_sen))

            ###

            obs_len = len(clean_text(cur_obs).split(' '))
            if obs_len > max_obs_len:
                max_obs_len = obs_len

            act_len = len(clean_text(rand_act).split(' '))
            if act_len > max_act_len:
                max_act_len = act_len
                
    file.close()

    ##avg_sen_len  = avg_sen_len / total_sents

    ##print("~~~~~~~", avg_sen_len, max_sen_len)

    ##print(largest_magnitude_q)

    pad_len = max_obs_len
    if max_act_len > max_obs_len:
        pad_len = max_act_len

    return data, pad_len, largest_magnitude_q ##max_sen_len 


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

def get_training_data(data, pad_len, q_scale, word_to_id, embedding_matrix):##max_obs_len, max_act_len, word_to_id, embedding_matrix): ##max_sen_len, word_to_id, embedding_matrix):
    lstm_dict = {}

    ##train_x = np.empty((len(data), max_sen_len, 50), "float32")
    
    ###

    obs_input = np.empty((len(data), pad_len, 50), "float32") #max_obs_len, 50), "float32")
    act_input = np.empty((len(data), pad_len, 50), "float32") ##max_obs_len, 50), "float32")

    ##smallest_q = 100
    
    
    q_vals = np.empty(len(data), "float32")

    for i in range(len(data)):
        print(i, "/", len(data))

        cur_state = data[i][0]
        rand_act = data[i][2]


        ##combined_str = cur_state + ' ' + rand_act

        q_score = float(data[i][3]) / q_scale ## data[i][4]

        ##q_score = float(format(q_score, ".4f"))

        ##if smallest_q > q_score and q_score > 0:
            ##smallest_q = q_score

        # if not (combined_str in lstm_dict):
        #     result = sentence_to_vect_sequence(combined_str, word_to_id, embedding_matrix) #sentence_to_vect(combined_str, word_to_id, embedding_matrix, lstm)
           
        #     lstm_dict[combined_str] = result
     
        # train_x[i] = pad_input(lstm_dict[combined_str], max_sen_len)
        # q_vals[i] = q_score

        ###

        if not (cur_state in lstm_dict):
            result = sentence_to_vect_sequence(cur_state, word_to_id, embedding_matrix)
            lstm_dict[cur_state] = result
        obs_input[i] = pad_input(lstm_dict[cur_state], pad_len)##max_obs_len)

        if not (rand_act in lstm_dict):
            result = sentence_to_vect_sequence(rand_act, word_to_id, embedding_matrix)
            lstm_dict[rand_act] = result
        q_vals[i] = q_score

        ##print(rand_act, lstm_dict[rand_act])
        ##print(rand_act)
        act_input[i] = pad_input(lstm_dict[rand_act], pad_len)##max_obs_len)

    ##print("0----------0", smallest_q)
    return (obs_input, act_input, q_vals), lstm_dict ##(train_x, q_vals), lstm_dict ##(lstm_train_x, lstm_train_y), lstm_dict, q_vals
    
## MAJORLY NEEDS LOOKED AT 
## ISSUE IS THAT I NEED TO PASS LSTM SEQUENCE OF WORD VECTORS (sentence) TO OUTPUT SENTENCE STATE VECTOR
# NEEDS TO FIGURE OUT HOW TO GET OUTPUT TO FIT THOUGH WHAT THE HELL
def sentence_to_vect_sequence(sentence, word_to_id, embedding_matrix):  
    ## sentence_vect = np.empty([1, 50, 1])
    word_ls = sentence.split(' ')
    word_vect_sequence = []
    for word in word_ls:
        if not word in word_to_id: # if we do not know a word then ignore it ## best solution?
            # print("---Vector for word not found:", word)
            # return np.array([])
            continue

        ##print(embedding_matrix[word_to_id[word]])

        ##word_vect_sequence.append(embedding_matrix[word_to_id[word]])
        word_vect_sequence.append(np.array(embedding_matrix[word_to_id[word]]).astype('float32'))

    ##print(word_vect_sequence)

    if word_vect_sequence == []:
        word_vect_sequence.append(np.array([0.0] * 50)) 
    
    ##print(word_vect_sequence)
    return np.array(word_vect_sequence)


def clean_sent_to_input(sentence, lstm_dict):
    if sentence in lstm_dict.keys():
        return lstm_dict.keys()
    
    return None

def pad_input(x_train, max_len):
    ##print(x_train.shape)
    while(len(x_train) < max_len):
        ##x_train[i].append([0] * 50)
        x_train = np.append(x_train, np.array([[0.0] * 50]), 0)

    ##print(x_train.shape)
    return x_train

def generate_net():
    # Hyper Params (need adjust)
    dense_nodes = 50
    learning_rate = 0.0005
    lstm_layer_dim = 50
    ##output_size = 1

    # print("Making Q-Net")
    model = Sequential()

    model.add(Masking(mask_value = 0.0))

    model.add(LSTM(lstm_layer_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(lstm_layer_dim, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(layers.Dense(dense_nodes, activation = 'relu'))

    model.add(AveragePooling1D())

    model.add(layers.Dense(dense_nodes)) ##, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(layers.Dense(dense_nodes)) ##, activation = 'relu'))
    model.add(BatchNormalization())

    model.add(layers.Dense(1)) #, activation = 'sigmoid'))
    
    model.compile(
        loss='huber', # 0.004
        ##loss='mse',
        ##loss='mean_squared_error', # 0.002
        ##loss='mean_squared_logarithmic_error', # 0.001
        ##loss='mean_absolute_error', 0.004
        
        ##optimizer=Adam(learning_rate = learning_rate),
        optimizer=RMSprop(learning_rate = learning_rate),
        ##optimizer='sgd',

        metrics=['accuracy'],   
    )

    return model


def generate_branched_net(state_input_shape, action_input_shape):
    # Hyper Params (need adjust)
    dense_nodes = 50
    learning_rate = 0.0005
    lstm_layer_dim = 10
    ##output_size = 1

    # Branch 1 (observation)
    state_input = Input(shape=state_input_shape)
    state_branch = Masking(mask_value = 0.0) (state_input)
    state_branch = LSTM(lstm_layer_dim, return_sequences=True) (state_branch)
    ##state_branch = Dropout(0.2) (state_branch)
    state_branch = LSTM(lstm_layer_dim, return_sequences=True) (state_branch)
    state_branch = Dropout(0.2) (state_branch)
    state_branch = Model(inputs=state_input, outputs=state_branch)

    # Branch 2 (action)
    action_input = Input(shape=action_input_shape)
    action_branch = Masking(mask_value = 0.0) (action_input)
    action_branch = LSTM(lstm_layer_dim, return_sequences=True) (action_branch)
    ##action_branch = Dropout(0.2) (action_branch)
    action_branch = LSTM(lstm_layer_dim, return_sequences=True) (action_branch)
    action_branch = Dropout(0.2) (action_branch)
    action_branch = Model(inputs=action_input, outputs=action_branch)
    
    # Combined Branch
    merging_layer = concatenate([state_branch.output, action_branch.output])

    combined_layers = AveragePooling1D() (merging_layer)
    combined_layers = Dense(dense_nodes, activation='relu') (combined_layers)
    combined_layers = Dense(dense_nodes, activation='relu') (combined_layers)
    combined_layers = Dense(1, activation='linear') (combined_layers)

    model = Model(inputs=[state_branch.input, action_branch.input], outputs=combined_layers)
    
    model.compile(
        loss='huber', # 0.004
        ##loss='mse',
        ##loss='mean_squared_error', # 0.002
        ##loss='mean_squared_logarithmic_error', # 0.001
        ##loss='mean_absolute_error', 0.004
        
        ##optimizer=Adam(learning_rate = learning_rate),
        optimizer=RMSprop(learning_rate = learning_rate),
        ##optimizer='sgd',

        metrics=['accuracy'],   
    )

    return model


def compile_corupus():
    dir_name = "data"
    files = os.listdir(dir_name)

    write_str = ""
    punctuation_to_remove = ['.', ':', '?', ',', ';', '"', "'", '|', '*', "'s"]

    for file in files:
        f = open(dir_name + '/' + file, 'r')

        for line in f:
            l_split = line.lower().split('$')[:3]
            l_split_tokens = []
            for i in range(len(l_split)):
                l_split_space = l_split[i].replace('|', ' ').split(' ')
                for j in range(len(l_split_space)):
                    clean_word = l_split_space[j]
                    for punct in punctuation_to_remove:
                        clean_word = clean_word.replace(punct, '')

                    clean_word = clean_word.strip()

                    if clean_word != '':
                        l_split_tokens.append(clean_word)
                

            l = ' '.join(l_split_tokens)
            ##print(l)

            ##print(l)
            write_str += l

    dest = open("GloVe-master/text.txt", 'w')
    dest.write(write_str)


def load_train_inf(obs_shape, act_shape):
    obs_train = np.loadtxt("data/-obs_train.txt").reshape(obs_shape)
    act_train = np.loadtxt("data/-act_train.txt").reshape(act_shape)
    q_train = np.loadtxt("data/-q_train.txt")

    return (obs_train, act_train, q_train)


def save_train_inf(training):
    print("--Saving obs_train of shape", training[0].shape)
    np.savetxt("data/-obs_train.txt", training[0].reshape(training[0].shape[0], -1))

    print("--Saving act_train of shape", training[1].shape)
    np.savetxt("data/-act_train.txt", training[1].reshape(training[0].shape[0], -1))

    print("--Saving q_train of shape", training[2].shape)
    np.savetxt("data/-q_train.txt", training[2])

##################################################
##data, max_obs_len, max_act_len = read_data("data") ##max_sen_len = read_data("data")
data, pad_len, q_scale_len = read_data("data")

if not exists("GloVe-Master/text.txt"):
    # generate_glove_embeddings(sentences)
    # NEED LOWERCASE NO PUNCT FROM ALL DATAFILES INTO text.txt in glove dir
    compile_corupus()
    print("needs GLOVE Embeddings! Then re-run")
elif(not exists("data/-obs_train.txt") or not exists("data/-act_train.txt") or not exists("data/-q_train.txt")):
    print("-Compiling Embeddings")
    word_to_id, embedding_matrix = compile_embeddings()

    print("-Getting Training Data")
    ##training, lstm_dict = get_training_data(data, max_obs_len, max_act_len, word_to_id, embedding_matrix) ##get_training_data(data, max_sen_len, word_to_id, embedding_matrix) ##, q_values = get_training_data(data, word_to_id, embedding_matrix)
    training, lstm_dict = get_training_data(data, pad_len, q_scale_len, word_to_id, embedding_matrix)

    print("-Saving Train Info")
    save_train_inf(training)

    print("--Training Data Generated! Re-run with shapes entered into code")

else:
    print("-Getting Training Info")
    word_to_id, embedding_matrix = compile_embeddings()
    training, lstm_dict = get_training_data(data, pad_len, q_scale_len, word_to_id, embedding_matrix)##max_obs_len, max_act_len, word_to_id, embedding_matrix)

    # obs_shape = (114516, 70, 50)
    # act_shape = (114516, 70, 50)
    # training = load_train_inf(obs_shape, act_shape)

    input_vects = [training[0], training[1]]
    q_values = training[2] 

    print("-Generating Net")
    ##model = generate_net()
    ###
    print(training[0][0].shape, training[1][0].shape)
    print(training[0][0])
    print('-')
    print(training[1][0])
    print('-')
    print(training[2][0])
    print('-')
    print(training[0][0][69][0] == 0.0)
    model = generate_branched_net(training[0][0].shape, training[1][0].shape) ##(len(data), max_obs_len, 50), (len(data), max_act_len, 50))

    print("-Training Net")
    epochs = 120
    batch_size = 400

    ##print(state_vects.shape, q_values.shape)
    
    ##model.fit(state_vects, q_values, epochs = epochs, batch_size = batch_size)

    ###

    model.fit(input_vects, q_values, epochs = epochs, batch_size = batch_size)



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
    
