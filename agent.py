"""
Agents for playing text-based games
"""

from math import sqrt
import random
import time
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
    def __init__(self, env: Environment):
        self.data = []
        self.discount = 0.99

    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        penultimate = False # Keep track if the next state will be the game end 

        cur_score = env.get_score() # Get the current score at this state
        cur_state = env.get_player_location() # Get the current state info

        valid_actions = env.get_valid_actions() # Get the list of valid actions form this state
        rand_act = random.choice(valid_actions) # Get a random valid action for the current state  

        cur_state = env.get_state() # Get the current state and save for switching back later

        _, _, done, _ = env.step(rand_act) # Move into the next state for info from env
        if done: # If the next state is the final state of this run, keep track 
            penultimate = True
        
        next_score = env.get_score()

        if not env.game_over: # If the run ends but we did not die, do not take penalty !!!!!!WILL BE FIXED WHEN CHAMBERS EDITS PLAY.PY!!!!!!
            next_score = cur_score

        score_diff = next_score - cur_score # Find out the score difference taking this action yields
        
        env.set_state(cur_state) 

        self.backpropogate(score_diff) # Backpropogate up the score w/ decay 
        self.data.append([env.get_player_location(), rand_act, next_score, score_diff]) # Append data entry 

        if penultimate: # If this is the last action we are taking record the data
            self.write_data()

        return rand_act # Return the action to take

    def backpropogate(self, score_change):
        for i in range(len(self.data)):
            index = len(self.data) - 1 - i
            self.data[index][2] += (score_change * (self.discount ** index))

    def write_data(self):
        data_str = ""
        for entry in self.data:
            data_str += str(entry[0]) + " + " + entry[1] + " == " + str(entry[2]) + " (" + str(entry[3]) + ")" + "\n"

        file = open("data_file.txt", "a") # CHANGE SECOND ARG TO a WHEN DONE TESTING OR w FOR OVERWRITE
        file.write(data_str)
        file.close()
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