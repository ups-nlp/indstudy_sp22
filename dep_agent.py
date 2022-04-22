"""
Original created 21.10.28
Recreated 22.2.6

@author Eric Markewitz, Penny Rowe, Danielle Dolan
"""

# Built-in modules
import random
import numpy as np
import re
import random
from operator import add
from operator import truediv

# Installed modules
from jericho import FrotzEnv
from config import VERBOSITY

import tensorflow as tf
from tensorflow.keras import models

# In-house modules
from agent import Agent


class DEPagent(Agent):

    def __init__(self):
        """
        Initialize the class by setting instance variables for movements,

        !!possibly weapons and enemies later!!
        """

        # Movements and their opposites
        self.movements = {'north': 'south', 'south': 'north',
                          'east': 'west', 'west': 'east',
                          'up': 'down', 'down': 'up',
                          'northwest': 'southeast', 'southeast': 'northwest',
                          'northeast': 'southwest', 'southwest': 'southeast',
                          'go': 'go', 'jump': 'jump'}

        # Number of past action to check
        self.PAST_ACTIONS_CHECK = 3

        self.vocab_vectors, self.word2id = embed_vocab()

        self.reconstructed_model = tf.keras.models.load_model('./NN/dm_nn')

    def mover(self, env: FrotzEnv, valid_actions: list, history: list) -> str:
        """
        @param FrotzEnv
        @param valid_actions
        @param history

        The mover will take the set of valid move actions and return one at random

        @return chosen_action: A string containing a move action
        """
        num_actions = len(valid_actions)

        if num_actions == 0:
            print("ERROR - NO VALID MOVE ACTIONS")
            return 'nva'

        # create empty list of size num actions
        action_weights = [0] * num_actions
        i = 0

        # Loop through actions and add weight to the indicies that line up w/ actions that have an action word in them
        while(i < num_actions):
            weight_val = 1
            curr_action = valid_actions[i]
            action_words = curr_action.split(' ')

            curr_obs = get_observation(env)

            for a_word in action_words:
                if a_word in curr_obs:
                    weight_val += 1
                    break  # only lets it double the weighting - no more

            action_weights[i] = weight_val
            i += 1

        print("action weights" + str(action_weights))
        total_weight = 0
        for weight in action_weights:
            total_weight += weight

        randomChoice = random.randint(0, total_weight)

        curr_weight = 0
        i = 0
        while(i < num_actions):
            curr_weight += action_weights[i]
            if(randomChoice <= curr_weight):
                chosen_action = valid_actions[i]
                return chosen_action
            i += 1

        # else:
            # return random.choice(valid_actions)

    def everything_else(self, env: FrotzEnv, valid_actions: list, history: list) -> str:
        """
        @param FrotzEnv
        @param valid_actions
        @param history

        The everything else will take all actions that aren't a movement and
        choose one at random

        @return chosen_action: A string containing a move actions
        """

        num_actions = len(valid_actions)

        if num_actions == 0:
            print("ERROR - NO VALID MOVE ACTIONS")
            return 'nva'

        if(VERBOSITY):
            print("EE actions " + str(valid_actions))
            print("Num EE actions: " + str(num_actions))

        action_weights = [0] * num_actions
        i = 0

        while(i < num_actions):
            weight_val = 1
            curr_action = valid_actions[i]
            action_words = curr_action.split(' ')

            curr_obs = get_observation(env)

            for a_word in action_words:
                if a_word in curr_obs:
                    weight_val += 1  # doubles the value every time there's a shared word
                    break  # max out at a double

            action_weights[i] = weight_val
            i += 1

        print("ee weights" + str(action_weights))
        total_weight = 0
        for weight in action_weights:
            total_weight += weight

        randomChoice = random.randint(0, total_weight)

        curr_weight = 0
        i = 0
        while(i < num_actions):
            curr_weight += action_weights[i]
            if(randomChoice <= curr_weight):
                chosen_action = valid_actions[i]
                return chosen_action
            i += 1

        # return random.choice(valid_actions)

    def take_action(self, env: FrotzEnv, history: list) -> str:
        """
        Takes in the history and returns the next action to take

        @param env, Information about the game state from Frotz
        @param history, A list of tuples of previous actions and observations

        @return action, A string with the action to take
        """

        # Get a list of all valid actions from jericho
        valid_actions = env.get_valid_actions()
        if(VERBOSITY):
            print("Valid actions " + str(valid_actions))

        # get sorted actions: in order: mover, everything_else
        sorted_actions = self.sort_actions(valid_actions)

        chosen_module = self.decision_maker(sorted_actions, env, history)

        action_modules = [self.mover, self.everything_else]

        # If the chosen module has no possible actions use the other one
        if len(sorted_actions[0]) == 0 and chosen_module == 0:
            chosen_module = 1
        elif len(sorted_actions[1]) == 0 and chosen_module == 1:
            chosen_module = 0

        action = action_modules[chosen_module](
            env, sorted_actions[chosen_module], history)

        current_obs = get_observation(env)

        if(current_obs == "with great effort you open the window far enough to allow entry"):
            if(VERBOSITY):
                print("WINDOW OPEN GOING EAST")
            return "west"

        # If the agent tries to jump, do literally any other move instead
        if action == 'jump':
            if(VERBOSITY):
                print("NO!!!! TRIED TO JUMP - HOPEFULLY AVERTED")

            # shuffle the list so you don't necessarily choose the same thing every time
            random.shuffle(sorted_actions[0])
            for move_action in sorted_actions[0]:
                if move_action != 'jump':
                    return move_action

            if(len(sorted_actions[1])):
                random.shuffle(sorted_actions[1])
                return sorted_actions[1][0]

        return action

    def decision_maker(self, sorted_actions: list, env: FrotzEnv, history: list) -> int:
        """
        Decide which choice to take.

        @param valid_actions
        @param environment

        Creates an embedding of all the words in the previous observation,
        runs that through a neural network that ranks how much we should use
        each of the modules. Then returns an int that represents the module
        with the highest value that has valid actions

        @return chosen_module: an integer of 0 or 1,
                               0 represents mover 1 represents everything_else
        """

        vector = self.create_observation_vect(env)
        np_vector = np.array([vector])

        prediction2DArr = self.reconstructed_model.predict(np_vector)

        #print("Prediction" + str(prediction))

        # 0 at the end because its a 2D array for some reason
        prediction = prediction2DArr[0][0]

        randVal = random.random()

        if(VERBOSITY):
            print("Decisionmaker NN value is (1 EE, 0 Move): " + str(prediction))
            print("Random value is: " + str(randVal))

        if(prediction < randVal):
            if(VERBOSITY):
                print("MOVER MODULE")
            return 0
        else:
            if(VERBOSITY):
                print("EVERYTHING ELSE")
            return 1

    def create_observation_vect(self, env: FrotzEnv) -> list:
        """
        Takes the gamestate and returns a vector representing the previous observation

        @param env: the current gamestate
        @return list: A normalized vector representing the previous observation
        """
        observation = get_observation(env)

        avg_vect = self.create_vect(observation)

        return(avg_vect)

    def create_vect(self, observation: str):
        """
        Takes an observation and returns a 50 dimensional vector representation of it

        @param str: a string containing an observation

        @return list: A list representing a 50 dimensional normalized vector
        """
        obs_split = observation.split(' ')

        vect_size = 50
        avg_vect = [0] * vect_size

        num_words = 0
        for word in obs_split:
            # Check if word is in vocab, if it is add it to the vector
            if(self.word2id.get(word) is not None):
                id = self.word2id.get(word)
                norm_vect = self.vocab_vectors[id]
                avg_vect = list(map(add, avg_vect, norm_vect))
                num_words += 1
            # else:
            #    if(VERBOSITY):
            #        print("Word not in the vocab: " + word)

        words = [num_words] * vect_size
        avg_vect = list(map(truediv, avg_vect, words))

        return(avg_vect)

    def sort_actions(self, valid_actions: list) -> list:
        """
        looks through all the valid actions and sorted them into mover
        or everything_else

        @param valid_actions

        @return sorted_actions, list of lists of sorted actions
        """
        mover_actions = []
        ee_actions = []

        for action in valid_actions:
            # check if action aligns with movements
            if action in self.movements:
                mover_actions.append(action)

            else:
                ee_actions.append(action)

        sorted_actions = [mover_actions, ee_actions]

        return sorted_actions


def embed_vocab() -> (list, dict):
    """
    Reads in the vocab and vector GloVemaster files in from the data folder.
    Returns a dictionary matching a word to an index and a list of the vectors

    @return list: A normalized list of vectors, one for each word
    @return dict: A dictionary with a word as a key and the id for the list as the value
    """
    with open("./data/vocab.txt", 'r') as f:
        # Read in the list of words
        words = [word.rstrip().split(' ')[0] for word in f.readlines()]

    with open("./data/vectors.txt", 'r') as f:
        # word --> [vector]
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            word = vals[0]
            vec = vals[1:]
            vectors[word] = [float(x) for x in vec]

    # Compute size of vocabulary
    vocab_size = len(words)
    word2id = {w: idx for idx, w in enumerate(words)}
    id2word = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[id2word[0]])

    # Create a numpy matrix to hold word vectors
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[word2id[word], :] = v

    # Normalize each word vector to unit length
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T

    return W_norm, word2id


def get_observation(env: FrotzEnv) -> str:
    curr_state = env.get_state()

    observation_index = 8
    gameState = curr_state[observation_index].decode()

    # Cleanup input
    onlyTxt = re.sub('\n', ' ', gameState)
    onlyTxt = re.sub('[,?()!.:;\'\"]', '', onlyTxt)
    onlyTxt = re.sub('\s+', ' ', onlyTxt)
    onlyTxt = onlyTxt.lower()

    # Remove the newline character
    onlyTxt = onlyTxt[:len(onlyTxt)-1]

    current_obs = onlyTxt

    return current_obs
