"""
Agents for playing text-based games
"""

from math import sqrt
import random
import time
from environment import *
from mcts_agent import best_child, tree_policy, default_policy, backup, dynamic_sim_len
from mcts_node import Node
from transposition_table import Transposition_Node
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
        self.root = Transposition_Node(env.get_world_state_hash(), None, None, env.get_valid_actions(), self.transposition_table)

        self.node_path.append(self.root)

        # This constant balances tree exploration with exploitation of ideal nodes
        self.explore_const = 1.0/sqrt(2)

        # The length of each monte carlo simulation
        self.simulation_length = 0

        # Maximum number of nodes to generate in the tree each time a move is made
        self.max_nodes = 200

        self.reward = AdditiveReward()

        self.history = {self.root.state}


    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""
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
        time_limit = 59

        # minimum number of nodes per simulation phase
        minimum = env.get_moves()*env.get_moves()

        #current state of the game. Return to this state each time generating a new node
        curr_state = env.get_state()
        while(seconds_elapsed < time_limit or count <= minimum):
            seconds_elapsed = time.time() - start_time
            if(count % 100 == 0): 
                print(count)
            # Create a new node on the tree
            new_node = tree_policy(self.root, env, self.explore_const, self.reward, self.transposition_table)
            # Determine the simulated value of the new node
            delta = default_policy(new_node, env, self.simulation_length, self.reward)
            # Propogate the simulated value back up the tree
            # create a hashSet of nodes already updated, so we don't update the same state twice
            #updated_set = set()
            if(delta < 0):
                print("delta value is ", delta)
            backup([(new_node, 0)], delta, set(), self.root)
            # reset the state of the game when done with one simulation
            env.reset()
            env.set_state(curr_state)
            count += 1

        print("Count: ", count)

        for child in self.root.get_children():
            child_sim_value = child.get_sim_value()
            child_visited = child.get_visited()
            print(child.get_prev_action(), ", count:", child_visited, ", value:", child_sim_value, "normalized value:", self.reward.select_action(env, child_sim_value, child_visited, None))

        ## Pick the next action
        self.root, score_dif = best_child(self.root, self.explore_const, env, self.reward, self.history, False)

        # update the history to include the new state
        self.history.add(self.root.state)

        self.node_path.append(self.root)

        ## Dynamically adjust simulation length based on how sure we are 
        self.max_nodes, self.simulation_length = dynamic_sim_len(self.max_nodes, self.simulation_length, score_dif)

        print("\n\n------------------ ", score_dif, self.max_nodes, self.simulation_length)

        return self.root.get_prev_action()