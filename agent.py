"""
Agents for playing text-based games
"""

from math import sqrt
import random
import time
import config
from environment import *
from mcts_agent import best_child, tree_policy, default_policy, backup
from mcts_node import MCTS_node, Node
from mcts_reward import BaselineReward

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
        print(env.get_valid_actions())
        print("Action: ")
        return input()


class MonteAgent(Agent):
    """"Monte Carlo Search Tree Player"""

    node_path = []

    def __init__(self, env: Environment, time_limit: int):
        # create root node with the initial state
        self.root = MCTS_node(None, None, env.get_valid_actions())

        self.node_path.append(self.root)

        # This constant balances tree exploration with exploitation of ideal nodes
        explore_const = 1.0/sqrt(2)
        self.reward = BaselineReward(explore_const)

        self.time_limit = time_limit


    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take using the Monte Carlo Search Algorithm"""                

        # number of iterations
        count = 0

        # time at sim start
        start_time = time.time()

        # how many seconds have elapsed since sim start
        seconds_elapsed = 0        

        #current state of the game. Return to this state each time generating a new node
        curr_state = env.get_state()

        if config.VERBOSITY > 1:
            print('[TAKE ACTION] Time limit: ', self.time_limit, ' seconds')
        
        while(seconds_elapsed < self.time_limit):
            seconds_elapsed = time.time() - start_time
           
            if config.VERBOSITY > 1:
               print('[TAKE ACTION] Root node is', str(self.root))
               print('[TAKE ACTION] Env valid actions', env.get_valid_actions())
               print('[TAKE ACTION] New actions', self.root.get_new_actions())


            # Create a new node on the tree
            new_node = tree_policy(self.root, env, self.reward)
            
            if config.VERBOSITY > 1:
                print('[TAKE ACTION] Chose a new node to expand', new_node.get_prev_action())
                print('[TAKE ACTION] Printing out parent again', str(self.root))
                print('[TAKE ACTION] Printing out selected child', str(new_node))

            # Determine the simulated value of the new node
            delta = default_policy(new_node, env)
            
            # Propogate the simulated value back up the tree
            backup(new_node, delta)
            # reset the state of the game when done with one simulation
            env.reset()
            env.set_state(curr_state)
            count += 1


        if config.VERBOSITY > 1:
            print('[TAKE ACTION] Number of iterations accomplished before time limit elapsed: ', count)
            
        if config.VERBOSITY > 0:
            print(env.get_valid_actions())
            for child in self.root.children:
                print(child.get_prev_action(), ", count:", child.visited, ", value:", child.sim_value, "normalized value:", self.reward.calculate_child_value(env, child, self.root))

        ## Pick the next action
        self.root = best_child(self.root, env, self.reward)

        self.node_path.append(self.root)

        return self.root.get_prev_action()