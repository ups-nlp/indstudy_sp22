"""
Agents for playing text-based games
"""

from math import sqrt
import random
import time
import os
from environment import *
from mcts_agent import best_child, tree_policy, default_policy, backup, dynamic_sim_len
from mcts_node import Node
from mcts_reward import AdditiveReward


class Agent:
    """Interface for an Agent"""

    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        raise NotImplementedError


class RandomAgent(Agent):
    """Agent randomly selects an action from list of valid actions"""

    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""

        valid_actions = env.get_valid_actions()
        return random.choice(valid_actions)


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
            self.write_data()
            print(self.total) #, cur_obs)
            self.count = 0

        #print(next_obs.split("\n")[0])

        return rand_act

        # dead_next = False # Keep track if the next state will be the game end 
        # win_next = False

        # cur_obs, _, _, _ = env.step('look')

        # cur_score = env.get_score() # Get the current score at this state
        # cur_state = env.get_player_location() # Get the current state info

        # rand_act = random.choice(valid_options) # Get a random valid action for the current state  

        # cur_state = env.get_state() # Get the current state and save for switching back later

        # next_obs, _, done, _ = env.step(rand_act) # Move into the next state for info from env
        # if env.game_over(): # If we are about to perish 
        #     dead_next = True
        # elif env.victory():
        #     win_next = True
        
        # next_score = env.get_score()

        # score_diff = next_score - cur_score # Find out the score difference taking this action yields
        
        # env.set_state(cur_state) 

        # self.backpropogate(score_diff) # Backpropogate down the score w/ decay

        # if not dead_next:
        #     self.data.append([cur_obs, next_obs, rand_act, next_score, next_score, "Reg", cur_state]) # Current obs, Next obs, Action, Potential future score total, Score at next state

        #     if win_next: # If we are about to win write data
        #         print("Victory")
        #         self.data[-1][5] = "Win"
        #         self.write_data()

        # else:
        #     print("Died")
        #     self.extra_states.append([cur_obs, next_obs, rand_act, next_score, next_score, "Los", cur_state]) # Current obs, Next obs, Action, Potential future score total, Score at next state
            
            
        #     self.revert_to_best(env)

        #     #self.look_ahead(env)
        #     rand_act = "look"

        #     # valid_actions.remove(rand_act)
        #     # if len(valid_actions) != 0: 
        #     #     rand_act = self.look_ahead(env, valid_actions)
        #     # else:
        #     #     self.write_data()

        # self.count += 1
        # self.total += 1
        # if self.count == 100:
        #     self.count = 0
        #     print("Step", self.total)

        # #print(rand_act)
        # return rand_act

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

        return (next_score - cur_score, next_state_status, next_obs)
        

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