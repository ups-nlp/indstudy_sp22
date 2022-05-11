import os
##from os.path import exists
##from telnetlib import SE

import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Dense, LSTM, AveragePooling1D, Input, BatchNormalization, Dropout, Masking, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy, Huber
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import HeUniform


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

            # obs_len = len(clean_text(cur_obs).split(' '))
            # if obs_len > max_obs_len:
            #     max_obs_len = obs_len

            # act_len = len(clean_text(rand_act).split(' '))
            # if act_len > max_act_len:
            #     max_act_len = act_len
                
    file.close()

    ##avg_sen_len  = avg_sen_len / total_sents

    ##print("~~~~~~~", avg_sen_len, max_sen_len)

    ##print(largest_magnitude_q)

    pad_len = max_obs_len
    if max_act_len > max_obs_len:
        pad_len = max_act_len

    return data, pad_len, largest_magnitude_q ##max_sen_len 


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


# def pad_input(x_train, max_len):
#     ##print(x_train.shape)
#     while(len(x_train) < max_len):
#         ##x_train[i].append([0] * 50)
#         x_train = np.append(x_train, np.array([[0.0] * 50]), 0)

#     ##print(x_train.shape)
#     return x_train


def state_action_pair_to_input(state, actions, word_to_id, embedding_matrix):
    processed_state = clean_text(state)
    processed_actions = [clean_text(action) for action in actions]

    max_act_len = max([len(act.split(' ')) for act in processed_actions])
    pad_len = max(len(processed_state.split(' ')), max_act_len)

    processed_state = process_input([processed_state], pad_len, word_to_id, embedding_matrix)[0]
    processed_actions = process_input(processed_actions, pad_len, word_to_id, embedding_matrix)

    return processed_state, processed_actions, processed_state.shape


def agent_data_to_input(data, word_to_id, embedding_matrix):
    #[[cur_obs, cur_score, act, next_obs, score_dif, next_state_status, next_acts, cur_state]] IN
    #[[cur_obs, act, score_dif, next_state_status, next_obs, next_actions]] OUT

    processed_data = []

    largest_sen_len = 0

    for i in range(len(data)):
        cur_obs, cur_score, act, next_obs, score_dif, next_state_status, next_actions, _, act_num, num_actions = data[i]

        ##print("0000000000000000", next_actions)
                                                                #VVVVVVVVV prev was cur score
        ##print(score_dif)
        processed_datum = [clean_text(cur_obs), clean_text(act), score_dif, next_state_status, clean_text(next_obs), [clean_text(action) for action in next_actions], 0, act_num, num_actions]

        # cur_obs_len = len(processed_datum[0].split(' '))
        # if cur_obs_len > largest_sen_len:
        #     largest_sen_len = cur_obs_len

        # act_len = len(processed_datum[1].split(' '))
        # if act_len > largest_sen_len:
        #     largest_sen_len = act_len

        # next_obs_len = len(processed_datum[4].split(' '))
        # if next_obs_len > largest_sen_len:
        #     largest_sen_len = next_obs_len

        # act_lens = [len(action.split(' ')) for action in processed_datum[5]]
        # if len(act_lens) != 0:
        #     max_act_len = max(act_lens)
        #     if  max_act_len > largest_sen_len:
        #         largest_sen_len = max_act_len
        ##else:
            ##print("What is going on here bruh", next_obs, '|', processed_data[5])

        processed_data.append(processed_datum)

           
    for i in range(len(processed_data)):
        processed_datum = processed_data[i]

        processed_datum[0] = process_input([processed_datum[0]], largest_sen_len, word_to_id, embedding_matrix)[0]
        
        processed_datum[1] = process_input([processed_datum[1]], largest_sen_len, word_to_id, embedding_matrix)[0]
        
        processed_datum[4] = process_input([processed_datum[4]], largest_sen_len, word_to_id, embedding_matrix)[0]
        
        processed_datum[5] = process_input(processed_datum[5], largest_sen_len, word_to_id, embedding_matrix)

        processed_datum[6] = processed_datum[0].shape

    return processed_data


def process_input(inputs, pad_len, word_to_id, embedding_matrix):
    processed_inputs = []
    for input in inputs:
        clean_input = input ##clean_text(input)

        input_vects = sentence_to_vect_sequence(clean_input, word_to_id, embedding_matrix)

        ##padded_input_vects = pad_input(input_vects, pad_len)

        processed_inputs.append(input_vects)

    return processed_inputs


def generate_branched_net(): ##, action_input_shape):
    ##print(state_input_shape)

    # Hyper Params (need adjust)
    ##dense_nodes = 64
    learning_rate = 0.001
    ##lstm_layer_dim = 10
    ##init = HeUniform()
    ##output_size = 1

    state_input = Input(shape=(None, 50))
    state_branch = LSTM(32, return_sequences=True) (state_input) ##(state_branch)
    state_branch = Dropout(0.2) (state_branch)
    state_branch = AveragePooling1D(pool_size = 2, padding = "same") (state_branch) ##
    state_branch = LSTM(16, return_sequences=True) (state_branch)
    state_branch = Dropout(0.2) (state_branch)
    state_branch = AveragePooling1D(pool_size = 2, padding = "same") (state_branch) ##
    state_branch = LSTM(8, return_sequences=False) (state_branch)
    state_branch = Model(inputs=state_input, outputs=state_branch)

    # Branch 2 (action)
    action_input = Input(shape=(None, 50))
    ##action_branch = Masking(mask_value = 0.0) (action_input)
    action_branch = LSTM(32, return_sequences=True) (action_input) ##(action_branch)
    action_branch = Dropout(0.2) (action_branch)
    action_branch = AveragePooling1D(pool_size = 2, padding = "same") (action_branch) ##
    action_branch = LSTM(16, return_sequences=True) (action_branch)
    action_branch = Dropout(0.2) (action_branch)
    action_branch = AveragePooling1D(pool_size = 2, padding = "same") (action_branch) ##
    action_branch = LSTM(8, return_sequences=False) (action_branch)
    action_branch = Model(inputs=action_input, outputs=action_branch)

    merging_layer = concatenate([state_branch.output, action_branch.output])

    # combined_layers = Dense(256, activation='relu') (merging_layer)
    # combined_layers = Dropout(0.2) (combined_layers)
    # combined_layers = Dense(256, activation='relu') (combined_layers)
    # combined_layers = Dense(256, activation='relu') (combined_layers)
    combined_layers = Dense(4, activation='relu') (combined_layers)
    combined_layers = Dense(2, activation='relu') (merging_layer) #(combined_layers)
    combined_layers = Dense(1, activation='linear') (combined_layers) 

    model = Model(inputs=[state_branch.input, action_branch.input], outputs=combined_layers)

    model.compile(
        loss='huber', # 0.004
        ##loss='mse',
        ##loss='mean_squared_error', # 0.002
        ##loss='mean_squared_logarithmic_error', # 0.001
        ##loss='mean_absolute_error', 0.004
        
        optimizer=Adam(learning_rate = learning_rate),
        ##optimizer=RMSprop(learning_rate = learning_rate),
        ##optimizer='sgd',

        metrics=['mean_squared_error'],   
    )

    ##print(model.summary())
    return model


def compile_corupus(dir_name):
    ##dir_name = "data"
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



##################################################


# data, pad_len, q_scale_len = read_data("data")



# print("-Getting Training Info")
# word_to_id, embedding_matrix = compile_embeddings()
# training, lstm_dict = get_training_data(data, pad_len, q_scale_len, word_to_id, embedding_matrix)##max_obs_len, max_act_len, word_to_id, embedding_matrix)

# input_vects = [training[0], training[1]]
# q_values = training[2] 

# model = generate_branched_net(training[0][0].shape, training[1][0].shape) ##(len(data), max_obs_len, 50), (len(data), max_act_len, 50))

# print("-Training Net")
# epochs = 120
# batch_size = 400

# model.fit(input_vects, q_values, epochs = epochs, batch_size = batch_size)

# print(model.summary())



