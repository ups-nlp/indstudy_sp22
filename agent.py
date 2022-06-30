"""
Agents for playing text-based games
"""

from math import sqrt, inf
import random
import time
from xml.etree.ElementTree import tostring
from environment import *
from mcts_agent import best_child, tree_policy, default_policy, backup
from transposition_table import Transposition_Node, get_world_state_hash
from mcts_reward import BaselineReward
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
        return random.choice(valid_actions), -1, -1

class HumanAgent(Agent):
    """Allows a human player"""

    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        print("Action: ")
        return input(), -1, -1


class MonteAgent(Agent):
    """"Monte Carlo Search Tree Player"""


    def __init__(self, env: Environment, time_limit: int, max_depth: int, explore_exploit: float):
    
        # Create the transposition table hashmap
        self.transposition_table = {}

        # Create and store the root node        
        valid_actions = env.get_valid_actions()
        score = env.get_score()
        state = get_world_state_hash(env.get_player_location(), valid_actions)
        self.root = Transposition_Node(state, None, None, valid_actions, self.transposition_table, score)        

        # This constant balances tree exploration with exploitation of ideal nodes
        self.explore_const = explore_exploit
        self.reward = BaselineReward(self.explore_const)

        self.time_limit = time_limit
        self.max_depth = max_depth
        self.alpha = 0.8


    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""        

        #current number of generated nodes
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
            
            if config.VERBOSITY > 1:
                print('\n\n==================== Count', count, '======================')
                print('[TAKE ACTION] Root node is', str(self.root))
                print('[TAKE ACTION] Root actions are', self.root.get_new_actions())
                print('[TAKE ACTION] Seconds elapsed ', seconds_elapsed, 'out of total of', self.time_limit)
            
            # Create a new node on the tree
            new_node, path = tree_policy(self.root, env, self.reward, self.transposition_table)

            if config.VERBOSITY > 1:
                print('[TAKE ACTION] Tree policy sampled path')
                total = ""
                for node in path:
                    if(node is not None):
                        total = total + str(node.get_prev_action()) + "->"
                print(total)

            # Determine the simulated value of the new node
            delta = default_policy(new_node, env, self.max_depth, self.alpha)

            # Propogate the simulated value back up the tree
            if(config.VERBOSITY > 1):
                print("delta value is ", delta)

            backup(path,delta)

            # reset the state of the game when done with one simulation
            env.reset()
            env.set_state(curr_state)

            if count > 100:
                print('Count is exploding', count)
                config.VERBOSITY = 2

            count += 1
            seconds_elapsed = time.time() - start_time



        
        if(config.VERBOSITY > 1):
            print('[TAKE ACTION] Number of iterations accomplished before time limit elapsed: ', count)

        if config.VERBOSITY > 0:
            print('Finished MCTS algorithm:')
            # for child in self.root.get_children():
            #     child_sim_value = child.get_sim_value()
            #     child_visited = child.get_visited()
            #     print(child.get_prev_action(), ", count:", child_visited, ", value:", child_sim_value)


        # Pick the action with highest average score
        # At this point, we do not factor in the number of times a node was chosen for expansion        
        best_action = None
        best_score = -inf        
        for child in self.root.get_children():
            if child.get_visited() == 0:
                if config.VERBOSITY > 0:
                    print(child.get_prev_action(), " count was 0")
                continue

            avg_score = child.get_sim_value()/child.get_visited()
            if config.VERBOSITY > 0:
                print(child.get_prev_action(), ", count:", child.get_visited(), ", value:", child.get_sim_value(), ", avg value:", avg_score)

            if avg_score > best_score:
                best_score = avg_score
                best_action = child   

        if best_action is None:
            exit("ALERT: At end of take_action(). The root node has 0 expanded children")

        self.root = best_action
        #self.root = best_child(self.root, env, self.reward)

        # Returning the chosen action, number of nodes added to tree, and number of states 
        return self.root.get_prev_action(), count, len(self.transposition_table)
