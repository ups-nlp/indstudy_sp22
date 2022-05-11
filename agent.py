"""
Agents for playing text-based games
"""

from cmath import inf, nan, tau
from math import sqrt
import random
import time
import os
from pygments import highlight


from torch import get_rng_state
from environment import *
from mcts_agent import best_child, tree_policy, default_policy, backup, dynamic_sim_len
from mcts_node import Node
from mcts_reward import AdditiveReward

from q_net import generate_branched_net
from q_net import agent_data_to_input
from q_net import compile_embeddings
from q_net import state_action_pair_to_input
from collections import deque

import numpy as np


class Agent:
    """Interface for an Agent"""

    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        raise NotImplementedError


class TestQNet(Agent):
    """Interface for an Agent"""

    def __init__(self, env: Environment):
        loaded_weights = [arr for arr in np.load("nets/qNet.npy", allow_pickle=True)]

        ##print(loaded_weights)

        self.model = generate_branched_net() 
        self.model.set_weights(loaded_weights)

        ##print(self.model.get_weights())
        
        # current_weights = self.model.get_weights()
        # for i in range(len(current_weights)):
        #     for j in range(len(current_weights[i])):
        #         print(type(loaded_weights[i][j]))
        
        self.word2id, self.id2word = compile_embeddings() # Get embeddings and dicts to translate
        ##print("INITIALIZED BAYBEEE")
        ##print(self.model.summary())

    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        valid_actions = env.get_valid_actions()
        ##print(env.step("look")[0])
        state_vects, act_vects, input_shape = state_action_pair_to_input(env.step("look")[0], valid_actions, self.word2id, self.id2word)

        ##self.adjust_model_size(input_shape)
        best_act_index = []
        best_act_q = -float(inf) 

        for i in range(len(act_vects)):
            ##print(np.array([state_vects]).shape)
            ##print(np.array([act_vects[i]]).shape)
            act_q = self.model.predict([np.array([state_vects]), np.array([act_vects[i]])])[0][0]

            if act_q != act_q: # Needed?
                act_q = -float(inf)

            print("-Q:", act_q, "For valid action:", valid_actions[i])

            if act_q > best_act_q:
                best_act_q = act_q
                best_act_index = [i]
            elif act_q == best_act_q:
                best_act_index.append(i)

        best_act = valid_actions[random.choice(best_act_index)]
        print("Best Estimated Action:", best_act)

        print("Enter Move:")
        return input()

    # def adjust_model_size(self, input_shape):
    #     if input_shape == self.input_shape:
    #         return

    #     self.input_shape = input_shape

    #     model = generate_branched_net(input_shape, 0)##input_shape)
        
    #     weights = self.model.get_weights()

    #     model.set_weights(weights)

    #     self.model = model
        


class RandomAgent(Agent):
    """Agent randomly selects an action from list of valid actions"""

    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""

        valid_actions = env.get_valid_actions()
        return random.choice(valid_actions)


class QTrainAgent(Agent):
    """Agent Plays the game and trains the QNet"""
    def __init__(self, env: Environment, save_net_dir, load_net = False):
        ##self.data_dir = data_dir

        # Q Learning vars
        self.data = deque(maxlen=1000) # State: obs, next_obs, action, q value, score, win/loss/regular
        self.gamma = 0.95 # discount for future scores
        self.epsilon = 1.0 # explore exploit
        self.epsilon_min = 0.1 # minimum value for epsilon
        self.epsilon_decay = 0.99999 # fraction at which epsilon decays ex: new_epsilon = epsilon * epsilon_decay
        self.tau = 0.125
        self.steps = 0
        
        # Model vars
        self.batch_size = 180

        self.target_net = generate_branched_net() # predictions?
        
        self.main_net = generate_branched_net() # desired predictions

        if load_net:
            self.target_net.set_weights([arr for arr in np.load(save_net_dir + "/qNet.npy", allow_pickle=True)])
            

            ## self.epsilon = SOMETHING

        self.word2id, self.id2word = compile_embeddings() # Get embeddings and dicts to translate 

        self.save_net_dir = save_net_dir

        # Game vars
        self.trials = 0
        self.moves = 0
        self.start_state = env.get_state()
        
        self.moves_since_last_point_change = 0
        self.action_max = 10
        self.last_state = (None, None)

        
    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""
    
        return self.next_move(env) #, valid_actions) # Return the action to take

    def next_move(self, env):
        cur_obs = env.step("look")[0]
        actions = env.get_valid_actions()
        next_act, act_num = self.evaluate_actions(cur_obs, actions) # Get the next action to take

        next_status = self.test_act(env, cur_obs, next_act, act_num) # Test the next action and record score/status. Get f we just died and are now reset or not

        if self.steps % 60 == 0: # No use training when all of the scores are 0 still
            self.batch_sample_train()
            self.update_main_weights()
            

        # If we died and just reset, just make this action 'look' 
        if next_status == -1 or self.moves_since_last_point_change > 100 or self.moves == 19:
            self.trials += 1
            print("Trial", self.trials, "Epsilon:", self.epsilon)
            next_act = "look"
            self.moves_since_last_point_change = 0
            self.moves = 0
            env.set_state(self.start_state)
            cur_obs = None
            actions = None
        ##elif next_status == 1:
            ##self.main_net.save(self.save_net_dir + "/qNet")

        if self.steps % 100 == 0:
            print("-Updating and Saving...")
            self.update_main_weights()
            ##self.main_net.save_weights(self.save_net_dir + "/qNet.h5")
            self.save_model_weights()

        self.moves += 1
        self.steps += 1

        ##print("Trial:", self.trials, "Moves:", self.moves, "Epsilon:", self.epsilon)

        self.last_state = (cur_obs, actions)

        return next_act # Return the action to the game

    def save_model_weights(self):
        np.save("nets/qNet", np.array(self.main_net.get_weights())) 
        ##print(self.main_net.get_weights())

    #
    def evaluate_actions(self, cur_obs, actions):
        self.epsilon *= self.epsilon_decay

        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

        if np.random.random() < self.epsilon:
            picked_action = np.random.choice(actions)
            return picked_action, actions.index(picked_action) 

        state_vects, action_vects, input_shape = state_action_pair_to_input(cur_obs, actions, self.word2id, self.id2word)
        
        ##self.adjust_model_size(self.target_net, input_shape, input_shape)

        highest_q = -float('inf')

        best_action_indices = []

        for i in range(len(action_vects)):
            q_val = self.main_net.predict([np.array([state_vects]), np.array([action_vects[i]])])[0][0]
            ##print(q_val)
            
            if q_val > highest_q:
                best_action_indices = [i]
            elif q_val == highest_q:
                best_action_indices.append(i)


        ##print(q_vals)

        # best_action_indices = []
        # for i in range(len(actions)):
        #     if highest_q < q_vals[i]:
        #         best_action_indices = [i]
        #     elif highest_q == q_vals[i]:
        #         best_action_indices.append(i)

    
        act_index = np.random.choice(best_action_indices)
        return actions[act_index], act_index 

    #
    def batch_sample_train(self):
        # If we do not have enough data to train on just move on
        if len(self.data) < self.batch_size:
            return 

        #
        rand_data = random.sample(self.data, self.batch_size)

        train_inf = agent_data_to_input(rand_data, self.word2id, self.id2word)

        total_loss = 0
        total_mse = 0

        for i in range(len(train_inf)):
            inf = train_inf[i]

            cur_state_vects, act_vect, reward, done, next_obs_vects, next_actions, input_shape, act_num, num_actions = inf

            if reward != 0:
               print("\n##########\n", rand_data[i][0], rand_data[i][2], reward)

            ##self.adjust_model_size(self.target_net, input_shape, num_actions)

            target_q = 0  

            if done:
                ##print("-End state", reward)
                target_q = reward
            else:
                ##highest_future_q = -float('inf')

                highest_future_q = 0

                q_vals = []

                ##print('-Cur state reward is:', reward)
                for act in next_actions:
                    q_val = self.main_net.predict([np.array([next_obs_vects]), np.array([act])])[0][0]
                    ##print(q_val)
                    q_vals.append(q_val)
                    ##print('--Potential future val:', q_val)

                ##print()

                highest_future_q = max(q_vals)
        
                if reward != 0:
                    print(highest_future_q * self.gamma)

                ##highest_future_q = max(highest_future_q, max_future_q)

                target_q = reward + (highest_future_q * self.gamma)
                ##print(target_q)

            ##print(type(target_q))

            ##print(np.array([cur_state_vects]).shape)
            ##print(np.array([act_vect]).shape)

            ##if reward != 0:
                ##print(target_q)

            fit_hist = self.target_net.fit([np.array([cur_state_vects]), np.array([act_vect])], np.array([target_q]), epochs = 1, batch_size = 1, verbose = 0).history

            if reward != 0:
                print(self.target_net.predict([np.array([cur_state_vects]), np.array([act_vect])]))

            total_loss += fit_hist['loss'][0]
            total_mse += fit_hist['mean_squared_error'][0]
            
        print("-Avg Loss:", total_loss / self.batch_size, "Avg MSE:", total_mse / self.batch_size)



    #
    def update_main_weights(self):
        target_weights = self.target_net.get_weights()
        ##print(target_weights, "\n``````````````````````````````````````````````````````````````````````````````")
        main_weights = self.main_net.get_weights()
        ##print(main_weights)
        
        new_main_weights = []

        for i in range(len(target_weights)):
            ##averaged_weights = (target_weights[i] + main_weights[i]) / 2
            ##print(type(target_weights[i]))
            ##print(adjusted_weights)
            new_main_weights.append(np.add((target_weights[i] * self.tau), (main_weights[i] * (1 - self.tau))))

        self.main_net.set_weights(new_main_weights)
        ##print(new_main_weights)
        
    #
    def test_act(self, env, cur_obs, act, act_num):

        cur_state = env.get_state()
        cur_score = env.get_score() + 10 ##+ (self.distance_deduction_multiplier * self.moves)

        score_dif, next_state_status, next_obs, next_acts, next_score = self.check_next_state(env, act)

        if score_dif == 0:
            self.moves_since_last_point_change += 1
        else:
            self.moves_since_last_point_change == 0

        if (next_obs, next_acts) != self.last_state: # attempt to prevent looping
            self.data.append([cur_obs, cur_score, act, next_obs, score_dif, next_state_status, next_acts, cur_state, act_num, len(env.get_valid_actions())])
        else:
            print("-dupe blocked")

        return next_state_status

    def check_next_state(self, env, action):
        next_state_status = 0

        cur_state = env.get_state()
        cur_score = env.get_score()

        next_obs = env.step(action)[0]
        next_score = env.get_score()
        next_actions = env.get_valid_actions()

        if env.game_over():
            next_state_status = -1
        elif env.victory():
            next_state_status = 1 
        
        env.set_state(cur_state)

        return next_score - cur_score, next_state_status, next_obs, next_actions, next_score ##+ (self.distance_deduction_multiplier * (self.moves + 1))



class CollectionAgent(Agent):
    """Agent randomly selects an action from list of valid actions and records progress for training data. Reverts to safe position on death"""
    def __init__(self, env: Environment, data_dir):
        self.data_dir = data_dir

        self.data = [] # State: obs, next_obs, action, q value, score, win/loss/regular
        self.extra_states = [] # Dead state: obs, next_obs, action, q value, score, win/loss/regular
        
        self.discount = 0.99 # Discount value

        self.count = 0 # Used for printing ever __ number of states
        self.total = 0 # How many states have we looked at

        #if not exists("data_file.txt"):
        #    open("data_file.txt", "a")

        # Get previous data in text file so it is not overwritten
        file = open("data/" + os.listdir("data")[-1], 'r') 
        self.prev_data = file.readlines()
        file.close()

    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        

        #valid_actions = env.get_valid_actions() # Get the list of valid actions from this state
        

        return self.look_ahead(env) #, valid_actions) # Return the action to take

    def look_ahead(self, env): #, valid_actions):
        valid_options = env.get_valid_actions()
        #while len(valid_options) == 0:
        #    self.revert_to_best(env)
        #    valid_options = env.get_valid_actions()

        cur_state = env.get_state()
        cur_score = env.get_score()
        cur_obs = env.step("look")[0]
        
        rand_act = random.choice(valid_options)

        score_dif, next_state_status, next_obs = self.check_next_state(env, rand_act)

        self.backpropogate(score_dif)

        state_status = "Status: " + str(next_state_status)
        
        ## CONDENSE STATEMENTS
        if next_state_status == -1:
            self.extra_states.append([cur_obs, next_obs, rand_act, cur_score, cur_score, state_status, cur_state])
            # cur_obs = self.data[self.revert_to_best(env)][0]
            # cur_obs = self.data[self.revert_to_weighted_random(env)][0]
            self.write_data()
            rand_act == ""
        elif next_state_status == 1:
            self.data.append([cur_obs, next_obs, rand_act, cur_score, cur_score, state_status, cur_state])
            self.write_data()
        else:
            self.data.append([cur_obs, next_obs, rand_act, cur_score, cur_score, state_status, cur_state])

        self.total += 1
        self.count += 1

        if self.count == 100:
            ##self.write_data()
            print(self.total) #, cur_obs)
            self.count = 0

        #print(next_obs.split("\n")[0])

        return rand_act


    def check_next_state(self, env, action): # Returns tuple (score_diff, is_not_game_over, next_state, next_obs)
        next_state_status = 0

        cur_state = env.get_state()
        cur_score = env.get_score()

        next_obs = env.step(action)[0]
        next_score = env.get_score()

        if env.game_over(): #or len(env.get_valid_actions()) == 0:
            next_state_status = -1
        elif env.victory():
            next_state_status = 1 
        
        env.set_state(cur_state)

        return (next_score - cur_score, next_state_status, next_obs) ## UNTUPLE
        

    def backpropogate(self, score_change):
        if score_change == 0.0:
            return
        for i in range(len(self.data)):
            index = len(self.data) - 1 - i
            self.data[index][3] += (score_change * (self.discount ** i))

    def revert_to_best(self, env):
        best_index = 0
        best_score = 0

        for i in range(len(self.data)): 
            if self.data[i][3] > best_score:
                best_score = self.data[i][3]
                best_index = i

        print("--Reverting to state with score", best_score)

        self.extra_states += self.data[best_index + 1:]
        self.data = self.data[0 : best_index + 1]

        env.set_state(self.data[best_index][6])
        env.step(self.data[best_index][2])
        
        #print("-", env.step('look')[0].split("\n")[0])

        return best_index

    def revert_to_weighted_random(self, env):
        selected_index = 0
        total_score = 0

        for i in range(len(self.data)): 
            total_score += self.data[i][3]

        random_score = random.random() * total_score

        score_so_far = 0
        for i in range(len(self.data)): 
            score_so_far += self.data[i][3]
            if score_so_far >= random_score:
                selected_index = i
                break


        print("--Reverting to state with score", self.data[selected_index][3], "chance of", random_score)

        self.extra_states += self.data[selected_index + 1:]
        self.data = self.data[0 : selected_index + 1]

        env.set_state(self.data[selected_index][6])
        env.step(self.data[selected_index][2])

        return selected_index

    def write_data(self):
        #data_str = ""
        data_ls = []
        for entry in self.data:
            #data_str += 
            data_ls.append(str(entry[0]).replace('\n', '|') + "$" + str(entry[1]).replace('\n', '|') + "$" + entry[2] + "$" + str(entry[3]) + "$" + str(entry[4]) + "$" + entry[5] + "\n")

        for entry in self.extra_states:
            #data_str += 
            data_ls.append(str(entry[0]).replace('\n', '|') + "$" + str(entry[1]).replace('\n', '|') + "$" + entry[2] + "$" + str(entry[3]) + "$" + str(entry[4]) + "$" + entry[5] + "\n")

        # if not exists("data_file.txt"):
        #     open("data_file.txt", "a")

        # file = open("data_file.txt", "r") 
        #data_ls = self.prev_data + data_ls
        
        #print("File Opened")
        
        #file.write(data_str)

        files = os.listdir(self.data_dir)

        file = open(self.data_dir + "/" + files[-1], 'a')#"w") # w for overwrite a for append
        
        file.writelines(data_ls)
        
        #file.writelines(data_ls)
        
        file.close()
        
        #print("File Closed")
        #print("Data Written to data_file.txt")
        


class HumanAgent(Agent):
    """Allows a human player"""

    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        print("Action: ")
        return input()



class MonteAgent(Agent):
    """"Monte Carlo Search Tree Player"""

    node_path = []

    def __init__(self, env: Environment, num_steps: int):
        # create root node with the initial state
        self.root = Node(None, None, env.get_valid_actions())

        self.node_path.append(self.root)

        # This constant balances tree exploration with exploitation of ideal nodes
        self.explore_const = 1.0/sqrt(2)

        # The length of each monte carlo simulation
        self.simulation_length = 15

        # Maximum number of nodes to generate in the tree each time a move is made
        self.max_nodes = 200

        self.reward = AdditiveReward()



    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        print("Action: ")
        #
        # Train the agent using the Monte Carlo Search Algorithm
        #

        #current number of generated nodes
        count = 0

        # time at sim start
        start_time = time.time()

        # how many seconds have elapsed since sim start
        seconds_elapsed = 0

        # loose time limit for simulation phase
        time_limit = 59

        # minimum number of nodes per simulation phase
        minimum = env.get_moves()*5

        #current state of the game. Return to this state each time generating a new node
        curr_state = env.get_state()
        while((seconds_elapsed < time_limit or count <= minimum)):
            seconds_elapsed = time.time() - start_time
            if(count % 10 == 0): 
                print(count)
            # Create a new node on the tree
            new_node = tree_policy(self.root, env, self.explore_const, self.reward)
            # Determine the simulated value of the new node
            delta = default_policy(new_node, env, self.simulation_length, self.reward)
            # Propogate the simulated value back up the tree
            backup(new_node, delta)
            # reset the state of the game when done with one simulation
            env.reset()
            env.set_state(curr_state)
            count += 1



        print(env.get_valid_actions())
        for child in self.root.children:
            print(child.get_prev_action(), ", count:", child.visited, ", value:", child.sim_value, "normalized value:", self.reward.select_action(env, child.sim_value, child.visited, None))

        ## Pick the next action
        self.root, score_dif = best_child(self.root, self.explore_const, env, self.reward, False)

        self.node_path.append(self.root)

        ## Dynamically adjust simulation length based on how sure we are 
        self.max_nodes, self.simulation_length = dynamic_sim_len(self.max_nodes, self.simulation_length, score_dif)

        print("\n\n------------------ ", score_dif, self.max_nodes, self.simulation_length)

        return self.root.get_prev_action()