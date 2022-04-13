"""
Agents for playing text-based games
"""

from math import sqrt
import random
import time
from xml.etree.ElementTree import tostring
from environment import *
from mcts_agent import best_child, tree_policy, default_policy, backup, dynamic_sim_len
from mcts_node import Node
from transposition_table import Transposition_Node, get_world_state_hash
from mcts_reward import AdditiveReward
import config

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

        
        # Create the transposition table hashmap
        self.transposition_table = {}

        # create root node with the initial state
        #self.root = Transposition_Node(None, None, env.get_valid_actions())
        
        state = get_world_state_hash(env.get_player_location(), env.get_valid_actions())
        self.root = Transposition_Node(state, None, None, env.get_valid_actions(), self.transposition_table)

        self.node_path.append(self.root)

        # This constant balances tree exploration with exploitation of ideal nodes
        self.explore_const = 1.0/sqrt(2)

        # The length of each monte carlo simulation
        self.simulation_length = 8

        self.reward = AdditiveReward()


    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        if(config.VERBOSITY > 0): 
            print("Action: ")
            print(env.get_valid_actions())
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
        time_limit = 90

        # minimum number of nodes per simulation phase
        minimum = len(env.get_valid_actions())*len(env.get_valid_actions())
        if(config.VERBOSITY > 0):
            print("count: ", count, ", minimum: ", minimum)

        #current state of the game. Return to this state each time generating a new node
        curr_state = env.get_state()
        #while(seconds_elapsed < time_limit or count <= minimum):
        while(seconds_elapsed < time_limit):
        #while(count < 100):
            seconds_elapsed = time.time() - start_time
            if(config.VERBOSITY > 0 and count % 50 == 0): 
                    print("Count is ",count)
            # Create a new node on the tree
            new_node, path = tree_policy(self.root, env, self.explore_const, self.reward, self.transposition_table)
            # Determine the simulated value of the new node
            delta = default_policy(new_node, env, self.simulation_length, self.reward)
            # Propogate the simulated value back up the tree
            if(config.VERBOSITY > 1):
                print("delta value is ", delta)
                total = ""
                for node in path:
                    if(node is not None):
                        total = total + str(node.get_prev_action()) + "->"
                print(total)
            backup(path,delta)
            # reset the state of the game when done with one simulation
            env.reset()
            env.set_state(curr_state)
            count += 1
        """
        print("Total count was: ", count)
        print("\nPrinting out transposition table:")
        for entry in self.transposition_table:
            print(entry, " : ", self.transposition_table[entry].toString())
        print("\nOptions for children:")

        """
        if(config.VERBOSITY > 0):
            print("seconds_elapsed: ", seconds_elapsed, ", time_limit: ", time_limit)
            print("count: ", count, ", minimum: ", minimum)
            for child in self.root.get_children():
                child_sim_value = child.get_sim_value()
                child_visited = child.get_visited()
                print(child.get_prev_action(), ", count:", child_visited, ", value:", child_sim_value, "normalized value:", self.reward.select_action(env, child_sim_value, child_visited, None))

        ## Pick the next action
        self.root, score_dif = best_child(self.root, self.explore_const, env, self.reward, False)

        self.node_path.append(self.root)

        ## Dynamically adjust simulation length based on how sure we are 
        #self.max_nodes, self.simulation_length = dynamic_sim_len(self.max_nodes, self.simulation_length, score_dif)

        if(config.VERBOSITY > 0):
            print("\n------------------ ", score_dif)#, self.max_nodes, self.simulation_length)

        return self.root.get_prev_action()