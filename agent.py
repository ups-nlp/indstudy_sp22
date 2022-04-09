"""
Agents for playing text-based games
"""

from math import sqrt
import random
import time
from environment import *
import mcts_agent
from simulation_length import simulation_length
#from mcts_agent import take_action, best_child, tree_policy, default_policy, backup
#from mcts_agent import mcts
from mcts_node import Node
from mcts_reward import AdditiveReward, DynamicReward
import multiprocessing_on_dill as multiprocessing
#from multiprocessing import Process,cpu_count
from multiprocessing_on_dill import Process, cpu_count, Queue, Lock
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

        self.simulation = simulation_length()

        self.node_path.append(self.root)

        # This constant balances tree exploration with exploitation of ideal nodes
        self.explore_const = 1.0/sqrt(2)

        #number of trees
        self.max_tree_count = cpu_count()
        self.tree_count = self.max_tree_count #1
        if(self.tree_count > self.max_tree_count):
            self.tree_count = self.max_tree_count


        #make trees
        #  the trees will continue to be built as the game continues
        self.tree_arr = [None]*self.tree_count
        for i in range(self.tree_count):
            self.tree_arr[i] = Node(None,None, env.get_valid_actions())


        #name the processes
        #UNUSED
        self.proc_names = [None]*self.tree_count
        for i in range(self.tree_count):
            self.proc_names[i] = "proc"+str(i)




        # The starting length of each monte carlo simulation
        self.simulation.initialize_agent(10)

        # Maximum number of nodes to generate in the tree each time a move is made
        self.max_nodes = 200

        #reward to use
        self.reward = DynamicReward()

        self.booster_threshold = 3



    def take_action(self, env: Environment, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        print("Action: ")
        print("possible actions: ")
        for act in env.get_valid_actions():
            print("\t",act)
 
        # time at sim start
        start_time = time.time()

        # how many seconds have elapsed since sim start
        seconds_elapsed = 0

        # loose time limit for simulation phase
        time_limit = 45

        # minimum number of nodes per simulation phase
        minimum = env.get_moves()*5



        #number of actions available from current spot
        num_actions = len(env.get_valid_actions())

        #create manager
        manager = multiprocessing.Manager()



        #create copies of the current environment for the trees
        #the environments will be stored in its own array
        self.env_arr = [None]*self.tree_count

        #create multiple simulation objects for the trees
        self.sim_list = [None]*self.tree_count
        for i in range(self.tree_count):
            self.sim_list[i] = simulation_length()
            self.sim_list[i].initialize_agent(self.simulation.get_length())

        #array will check to see if processes have returned cleanly
        self.has_returned = [True]*self.tree_count
        new_state = env.get_state()
        for i in range(self.tree_count):
            self.env_arr[i] = env.copy()
            self.env_arr[i].set_state(new_state)

        #create a queue for every tree. This queue will hold 
        # 1) a score dictionary
        # 2) a count dictionary
        # 3) the root to the tree
        # 4) the simulation object for that tree
        proc_queues = [None]*self.tree_count
        for i in range(self.tree_count):
            proc_queues[i] = Queue(4)
            proc_queues[i].put({})
            proc_queues[i].put({})
            proc_queues[i].put(self.tree_arr[i])
            proc_queues[i].put(self.sim_list[i])





        #set boolean that tells the trees to stop expanding when the time is up
        timer = multiprocessing.Value("i",0)

        #counter that holds how many of the trees have returned
        procs_finished = multiprocessing.Value("i",0)

        #lock so that only one tree can update counter at a time
        proc_lock = Lock()

        #send off different trees with their dicts and environments
        if __name__=="agent":
            print("spinning threads \n")
            procs = []
            for i in range(self.tree_count):
                #spin off a new process to take_action and append to processes list
                proc = Process(name = self.proc_names[i], target = mcts_agent.take_action, args = (proc_queues[i],self.env_arr[i],self.explore_const,self.reward,timer,procs_finished,proc_lock,))
                procs.append(proc)
                proc.start()

        #set main to sleep until the time limit runs out. In this time, the processes are building their trees
        time.sleep(time_limit)
        print("ending timer")

        #when the timer runs out, set the timer value to 1 so the threads start to return
        with timer.get_lock():
            timer.value = 1
        #time.sleep(1)

        #wait for all the processes to exit the while loop and restock the queues before joining
        while procs_finished.value < self.tree_count:
            time.sleep(1)
        print("joining processes")


        #store the score dictionaries and count dictionaries from the processes
        tree_scores = [None]*self.tree_count
        tree_counts = [None]*self.tree_count


        #join the trees
        for i in range(self.tree_count):
            proc = procs[i]

            #grab the shared queue and place the dictionaries into the lists
            que = proc_queues[i]
            tree_scores[i] = que.get()
            tree_counts[i] = que.get()
            self.tree_arr[i] = que.get()
            print("tree size for subtree: ",self.tree_arr[i].subtree_size)
            proc.join(5)

            #if the process doesn't exit cleanly, update its has_returned value
            if proc.exitcode is None:
                proc.terminate()
                proc.join()
                self.has_returned[i] = False

        #grab the best action, and the dictionaries that store the action and the counts 
        best_action, action_values, action_counts = self.best_shared_child(env, tree_scores, tree_counts)

        #for each action, print the calculated score and the total count
        for act in env.get_valid_actions():
            if act not in action_values.keys() or action_counts.get(act)==0:
                print("act ",act,"not explored.")
            else:
                 print("ACT ",act,": score = ",action_values.get(act)/action_counts.get(act),", count = ",action_counts.get(act))
        print("best child: ",best_action, "\n")

        for i in range(self.tree_count):
            #if the tree had to be forcibly joined
            if not self.has_returned:
                continue
            self.tree_arr[i] = self.tree_arr[i].get_child(best_action)


        self.node_path.append(self.root)

        return best_action




    def best_shared_child(self, env:Environment, tree_scores, tree_counts):
        """
        Calculates the best child based on the combined result of each tree

        The best action, a, will have the highest value Q(a) based on the following formula:
        Q(a) = sum_over_all_trees(score(a)*count(a))/sum_over_all_trees(count(a))
        This function calculates the Q(a) for each action from the current root and returns the 
        action with the highest value.

        Keyword arguments:
        env: game environment
        tree_scores: list of all the score dictionaries
        tree_counts: list of all the count dictionaries
        Return highest action, and the dictionaries containing combined scores and counts
        """
        #store the scores and counts for the actions accumulated over all trees
        action_score_dict = {}
        action_count_dict = {}

        #initialize to 0 score, 0 count
        for act in env.get_valid_actions():
            action_score_dict[act] = 0
            action_count_dict[act] = 0
        
        for i in range(self.tree_count):
            #if the tree has to be forcibly joined, don't use its data
            if not self.has_returned[i]:
                print("\t \t \t URGENT: tree ",i," has not returned")
                continue
            #grab the dictionaries for the tree
            score_dict = tree_scores[i]
            count_dict = tree_counts[i]
            for act in env.get_valid_actions():
                #if this action was not explored on this tree, continue
                if act not in score_dict.keys():
                    continue
                #update the accumulated dictionary with the values from this tree
                temp_score = action_score_dict[act]
                temp_count = action_count_dict[act]
                #if action_count_dict[act]>0 and temp_score/temp_count+self.booster_threshold <= (score_dict[act]/(count_dict[act]+1)):
                #    temp_score += self.booster_threshold
                #action_score_dict[act] = (score_dict[act]*count_dict[act])+temp_score
                action_score_dict[act] = score_dict[act]+temp_score


                action_count_dict[act] = count_dict[act]+temp_count
        return self.calculate_action_values(action_score_dict, action_count_dict), action_score_dict, action_count_dict


    def calculate_action_values(self, score_dict, count_dict):
        """
        Calculates the highest scoring action.

        Keyword arguments:
        score_dict: dictionary containing the actions and the accumulated scores over all trees
        count_dict: dictionary containing the actions and the accumulated counts over all trees

        return: highest scoring action
        """
        max_score = 0
        max_act = None
        for act in score_dict.keys():
            #if the action was not explored at all, it does not have the highest score
            if count_dict[act] == 0:
                continue

            #if the action has the highest score so far, update value
            if score_dict[act]/count_dict[act] >= max_score:
                max_score = score_dict[act]/count_dict[act]
                max_act = act
        return max_act

 


    
    



