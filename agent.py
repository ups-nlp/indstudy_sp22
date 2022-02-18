"""
Agents for playing text-based games
"""

from math import sqrt
import random
import time
from xmlrpc.client import Boolean
from jericho import FrotzEnv
import mcts_agent
from mcts_agent import best_child, tree_policy, default_policy, backup, dynamic_sim_len
from mcts_node import Node
from mcts_reward import AdditiveReward
import multiprocessing
from multiprocessing import Process

class Agent:
    """Interface for an Agent"""

    def take_action(self, env: FrotzEnv, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        raise NotImplementedError


class RandomAgent(Agent):
    """Agent randomly selects an action from list of valid actions"""

    def take_action(self, env: FrotzEnv, history: list) -> str:
        """Takes in the history and returns the next action to take"""

        valid_actions = env.get_valid_actions()
        return random.choice(valid_actions)

class HumanAgent(Agent):
    """Allows a human player"""

    def take_action(self, env: FrotzEnv, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        print("Action: ")
        return input()

class MonteAgent(Agent):
    """"Monte Carlo Search Tree Player"""

    node_path = []

    def __init__(self, env: FrotzEnv, num_steps: int):
        # create root node with the initial state
        self.root = Node(None, None, env.get_valid_actions())

        self.node_path.append(self.root)

        # This constant balances tree exploration with exploitation of ideal nodes
        self.explore_const = 1.0/sqrt(2)

        #number of trees
        self.tree_count = 2
        self.max_tree_count = multiprocessing.cpu_count()
        if(self.tree_count > self.max_tree_count):
            self.tree_count = self.max_tree_count


        #make trees
        self.tree_arr = [None]*self.tree_count
        for i in range(self.tree_count):
            self.tree_arr[i] = Node(None,None, env.get_valid_actions())

        #create environments for each of these trees
        self.env_arr = [None]*self.tree_count


        # The length of each monte carlo simulation
        self.simulation_length = 15

        # Maximum number of nodes to generate in the tree each time a move is made
        self.max_nodes = 200

        self.reward = AdditiveReward()



    def take_action(self, env: FrotzEnv, history: list) -> str:
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



        #number of actions available from current spot
        num_actions = len(env.get_valid_actions)

        #create lists to hold the dictionaries that will be sent to each tree:
        #   score_list holds a list of dictionaries (once dict / tree) where the key is 
        #   the action and the value is the normalized score for that action
        #
        #   count_list holds a list of dictionaries (one dict / tree) where the key is
        #   the action and the value is the number of times that action has been
        #   explored from the root of the tree
        score_list = [None]*self.tree_count
        count_list = [None]*self.tree_count
        for i in range(self.tree_count):
            score_list[i] = {}
            count_list[i] = {}

        #create copies of the current environment for the trees
        for i in range(self.tree_count):
            self.env_arr = env.copy()

        #set boolean that tells the trees to stop expanding when the time is up
        timer = True

        #send off different trees with their dicts and environments
        #also get a stopping boolean, and their own random seed generator so they randomly pick objects
        #FILL IN
        if __name__=="__main__":
            procs = []
            #self, root, env: FrotzEnv, explore_exploit_const, reward_policy, score_dict, count_dict, timer
            for i in range(self.tree_count):
                proc = Process(target = mcts_agent.take_action, args = (self.tree_arr[i],self.env_arr[i],self.explore_const,self.reward,score_list[i],count_list[i],timer,))
                procs.append(proc)
                proc.start()
            while((seconds_elapsed < time_limit or count <= minimum)):
                seconds_elapsed = time.time() - start_time
            timer = False
            for proc in procs:
                proc.join()


        #current state of the game. Return to this state each time generating a new node
        curr_state = env.get_state()
       
        ## Pick the next action
        best_action= self.best_shared_child(env,score_list,count_list,num_actions)

        #send action to all trees to take step and make the new root
        for tr in self.tree_arr:
            tr.root = best_action
        #FILL IN

        self.node_path.append(self.root)

        ## Dynamically adjust simulation length based on how sure we are 
        #self.max_nodes, self.simulation_length = dynamic_sim_len(self.max_nodes, self.simulation_length, score_dif)

        #print("\n\n------------------ ", score_dif, self.max_nodes, self.simulation_length)

        return self.root.get_prev_action()

    def best_shared_child(self,env: FrotzEnv, score_list:list, count_list:list, num_actions:int):
        #take in list of the values received from each action and calculate the score
        action_values = {}*num_actions
        action_counts = {}*num_actions
        max = 0
        max_act = ""
        for act in env.get_valid_actions:
            action_values[act], action_counts[act] = self.calculate_action_values(score_list, count_list, act)
            if action_values[act]/action_counts[act] > max:
                max = action_values[act]/action_counts[act]
                max_act = act
        return max_act

    def calculate_action_values(self, score_list:list, count_list:list, act):
        score = 0
        count = 0
        for i in range(len(score_list)):
            score_dict = score_list[i]
            count_dict = count_list[i]
            score = score + (score_dict.get(act)*count_dict.get(act))
            count = count + count_dict.get(act)
        return score, count
    



