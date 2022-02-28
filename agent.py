"""
Agents for playing text-based games
"""

from math import sqrt
import random
import time
from environment import *
import mcts_agent
#from mcts_agent import take_action, best_child, tree_policy, default_policy, backup
#from mcts_agent import mcts
from mcts_node import Node
from mcts_reward import AdditiveReward
import multiprocessing_on_dill as multiprocessing
#from multiprocessing import Process,cpu_count
from multiprocessing_on_dill import Process, cpu_count
#from pathos.pools import ParallelPool
#import dill as pickle

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
        # create root node with the initial state
        self.root = Node(None, None, env.get_valid_actions())

        self.node_path.append(self.root)

        # This constant balances tree exploration with exploitation of ideal nodes
        self.explore_const = 1.0/sqrt(2)

        #number of trees
        self.tree_count = 1
        self.max_tree_count = cpu_count()
        if(self.tree_count > self.max_tree_count):
            self.tree_count = self.max_tree_count


        #make trees
        self.tree_arr = [None]*self.tree_count
        for i in range(self.tree_count):
            self.tree_arr[i] = Node(None,None, env.get_valid_actions())


        self.proc_names = [None]*self.tree_count
        for i in range(self.tree_count):
            self.proc_names[i] = "proc"+str(i)


        #make objects
        #self.object_arr = [None]*self.tree_count
        #for i in range(self.tree_count):
        #    self.object_arr[i] = mcts()

        #create environments for each of these trees
        self.env_arr = [None]*self.tree_count


        # The length of each monte carlo simulation
        self.simulation_length = 50

        # Maximum number of nodes to generate in the tree each time a move is made
        self.max_nodes = 200

        self.reward = AdditiveReward()


    class threadTimer:
        def __init__(self):
            self.timer = True

        def end(self):
            self.timer = False





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
        time_limit = 10

        # minimum number of nodes per simulation phase
        minimum = env.get_moves()*5



        #number of actions available from current spot
        num_actions = len(env.get_valid_actions())

        #create lists to hold the dictionaries that will be sent to each tree:
        #   score_list holds a list of dictionaries (once dict / tree) where the key is 
        #   the action and the value is the normalized score for that action
        #
        #   count_list holds a list of dictionaries (one dict / tree) where the key is
        #   the action and the value is the number of times that action has been
        #   explored from the root of the tree
        score_list = [None]*self.tree_count
        count_list = [None]*self.tree_count
        print("tree count = ",self.tree_count)
        for i in range(self.tree_count):
            score_list[i] = {}
            count_list[i] = {}

        #create copies of the current environment for the trees
        for i in range(self.tree_count):
            #CHANGE TO COPY
            self.env_arr[i] = env



        #set boolean that tells the trees to stop expanding when the time is up
        #timer = self.threadTimer()
        timer = multiprocessing.Value("i",0)

        #send off different trees with their dicts and environments
        #also get a stopping boolean, and their own random seed generator so they randomly pick objects
        #FILL IN
        if __name__=="agent":
            print("spinning threads \n")
            #pool = ParallelPool(nodes=self.tree_count)
            #explore_list = [self.explore_const]*self.tree_count
            #sim_len_list = [self.simulation_length]*self.tree_count
            #reward_list = [self.reward]*self.tree_count
            #timer_list = [timer]*self.tree_count
            #pool.map(mcts_agent.take_action, self.tree_arr,self.env_arr,explore_list,sim_len_list,reward_list,score_list,count_list,timer_list)
            #root, env: Environment, explore_exploit_const, simulation_length, reward_policy, score_dict, count_dict, timer
            procs = []
            for i in range(self.tree_count):
                proc = Process(name = self.proc_names[i], target = mcts_agent.take_action, args = (self.tree_arr[i],self.env_arr[i],self.explore_const,self.simulation_length,self.reward,score_list[i],count_list[i],timer,))
                procs.append(proc)
                #proc.run()
                proc.start()
        else:
            print("not spinning threads \n")
            print("name = ",__name__)

        #while((seconds_elapsed < time_limit)):
            #if (0 == seconds_elapsed % 5):
            #    print("waiting...")
         #   seconds_elapsed = time.time() - start_time
        time.sleep(time_limit)
        print("ending timer")
        with timer.get_lock():
            timer.value = 1
        print("joining processes")

        for proc in procs:
            proc.join(1)
            if proc.is_alive():
                proc.terminate()
                proc.join()
                print("proc forcibly joined")
            print("proc joined: ",proc.is_alive())


        #current state of the game. Return to this state each time generating a new node
        curr_state = env.get_state()
       
        ## Pick the next action
        best_action= self.best_shared_child(env,score_list,count_list,num_actions)
        print("best child: ",best_action, "\n")

        #send action to all trees to take step and make the new root
        for tr in self.tree_arr:
            tr.root = best_action
        #FILL IN

        self.node_path.append(self.root)

        ## Dynamically adjust simulation length based on how sure we are 
        #self.max_nodes, self.simulation_length = dynamic_sim_len(self.max_nodes, self.simulation_length, score_dif)

        #print("\n\n------------------ ", score_dif, self.max_nodes, self.simulation_length)

        return self.root.get_prev_action()

    def run_mult_args(self, i, env:Environment, score_list:list, count_list:list, timer):
        return mcts_agent.take_action(self.tree_arr[i],env,self.explore_const,self.simulation_length,self.reward,score_list[i],count_list[i],timer)

    def best_shared_child(self,env: Environment, score_list:list, count_list:list, num_actions:int):
        #take in list of the values received from each action and calculate the score
        action_values = {}
        action_counts = {}
        max = 0
        max_act = random.choice(env.get_valid_actions())
        for act in env.get_valid_actions():
            action_values[act], action_counts[act] = self.calculate_action_values(score_list, count_list, act)
            if action_counts[act] > 0 and action_values[act]/action_counts[act] > max:
                print("updating max. prev max = ",max_act,", new act = ",act,"\n")
                max = action_values[act]/action_counts[act]
                max_act = act
        return max_act

    def calculate_action_values(self, score_list:list, count_list:list, act):
        score = 0
        count = 0
        print("calculating action values for ",act,"\n")
        for i in range(len(score_list)):
            if act not in score_list[i].keys():
                continue

            score_dict = score_list[i]
            count_dict = count_list[i]
            score = score + (score_dict.get(act)*count_dict.get(act))
            count = count + count_dict.get(act)
        print("total score = ",score," total count = ",count,"\n")
        return score, count
    



