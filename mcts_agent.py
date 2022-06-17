"""
An implementation of the UCT algorithm for text-based games
"""
from math import floor, inf
from platform import architecture
import random
import os
from xmlrpc.client import Boolean
from environment import *
from mcts_node import Node, MCTS_node
from mcts_reward import *
ACTION_BOUND = .01
SIM_SCALE = .1
THRESHOLD = 1


class Foo:
    pass
foo=Foo()

def exit_handler():
    x = 2
    # print("finished process:"+str(os.getpid()))


def workInitialize():
        #print("initialize:"+str(os.getpid()))

        from multiprocessing.util import Finalize
        #create a Finalize object, the first parameter is an object referenced 
        #by weakref, this can be anything, just make sure this object will be alive 
        #during the time when the process is alive 

        Finalize(foo, exit_handler, exitpriority=0)

def take_action(queue_list, env: Environment, explore_exploit_const, reward_policy, timer, procs_finished, nodes_generated, simulation_length, alpha, lock):
#def take_action(tree, sim, env: Environment, explore_exploit_const, reward_policy):

    """
    Take action continuously expands and explores the tree from the root passed in.
    It runs until a timer from the main thread tells it to stop, or until it calculates 10,000 nodes.

    Keyword arguments:
    queue_list: shared queue that stores the score dictionary, count dictionary, and root
    env: game environment
    explore_exploit_const: constant that determines balance between exploring new path versus taking old path
    simulation_length: length of the simulation to run from each new node
    alpha: value to discount score by as you walk through the simulation
    reward_policy: determines which reward policy to use
    timer: shared timer will set to false when the time is up, and this function will return
    procs_finished: shared integer that holds the number of processes that have returned from take_action
    lock: locks procs_finished so only one process can edit at a time
    """

    workInitialize()

    #set random seed with time stamp + thread number

    #get the dictionaries off of the multiprocessing queue
    #print(env.get_valid_actions())
    score_dict = queue_list.get()
    count_dict = queue_list.get()
    root = queue_list.get()
    # simulation = queue_list.get()
    score_dict ={}
    count_dict = {}
    # root = tree
    # simulation = sim
    curr_state = env.get_state()
    count = 0
    action = None
    # sim_length_scale = 1
    num_children = len(env.get_valid_actions())
    total_scoring_states = 0

    #if the children of the root have been thoroughly explored and there are no 
    #obvious score differences, adjust the simulation length before exploring further down
    #this tree

    #timer.value==0 and 
    #while(timer.value==0 and count < 20000):
    while(timer.value == 0):

        count = count+1

        #create a new node on the tree and store the action taken to get there

        new_node,action = tree_policy(root, env, explore_exploit_const, reward_policy)

        #update the size of the tree
        update_tree(new_node)
        # Determine the simulated value of the new node
        adjust_scoring_states = 0
        delta, adjust_scoring_states = default_policy(new_node, env, simulation_length, alpha)

        total_scoring_states += adjust_scoring_states

        # Propogate the simulated value back up the tree
        backup(new_node, delta)

        # reset the state of the game when done with one simulation
        env.set_state(curr_state)

        #update the shared table 
        if action in count_dict.keys():
            new_count = count_dict[action]+1
        else:
            new_count = 1
        score_dict[action] = root.get_child(action).get_sim_value()
        count_dict[action] = new_count

    # print(score_dict)
    # print(count_dict)
    #print("TREE TOTAL RUNS = ",count," SIM LENGTH = ",simulation.get_length()," SCORING STATES = ",total_scoring_states)
    # if total_scoring_states < THRESHOLD and root.get_prev_action() is not None and count >= num_children*2:
    #     # print("incrementing sim scale")
    #     simulation.adjust_sim_length(1+SIM_SCALE)
    #     # print("adjusted sim length: ",simulation.get_length())

    # elif calc_score_dif(root) <=ACTION_BOUND and root.get_prev_action() is not None and root.subtree_size >= num_children*2 and total_scoring_states > THRESHOLD:
    #     # print("decrementing sim score")
    #     sim_length_scale = 1/num_children
    #     #simulation_length = simulation_length*sim_length_scale
    #     simulation.adjust_sim_length(sim_length_scale)
    #     if simulation.get_length() < 10:
    #         simulation.adjust_sim_length(1/sim_length_scale)
    #     # print("sim length:", simulation.get_length())


    #after leaving the action sequence, place the dictionaries back on the shared queue
    queue_list.put(score_dict)
    queue_list.put(count_dict)
    # queue_list.put(simulation)
    queue_list.put(root)
    # #if the tree is small enough to be parsed, place the entire tree on the queue
    # if root.get_subtree_size() <=9:
        
    # #otherwise, remove the largest child and put the tree on the queue in parts
    # else:
    #     print("parsing tree")
    #     child = get_largest_child(root)
    #     root.remove_child(child)
    #     child.parent = None
    #     queue_list.put(root)
    #     queue_list.put(child)
    #     print("put tree on queue")


    lock.acquire()
    nodes_generated.value += count
    procs_finished.value +=1
    lock.release()
    queue_list.close()
    return    
   


def get_largest_child(node):
    max_size = 0
    max_chil = None
    for chil in node.get_children():
        if chil.get_subtree_size() > max_size:
            max_size = chil.get_subtree_size()
            max_chil = chil
    # print("retirning largest chil")
    return max_chil




def calc_score_dif(node):
    """
    Calculates the difference between the smallest scoring child and the largest scoring child

    This function sorts the children of the input node into a list ordered on the children's scores,
    then calculates the difference between the first and last element.

    Keyword arguments:
    node: the node to calculate the child difference on
    Return the difference
    """
    child_list = node.get_children()

    #If there are no children, do not adjust the simulation length
    if len(child_list)==0:
        return ACTION_BOUND+1

    #create a list and fill with the simulated values from each child
    child_val_list = [0]*len(child_list)
    for i in range(len(child_list)):
        child_val_list[i] = child_list[i].sim_value

    #sort the list
    child_val_list.sort()

    #calculate the difference between the first and last element
    largest_diff = abs(child_val_list[len(child_val_list)-1]) - child_val_list[0]
    return largest_diff


def update_tree(node):
    """
    Update the size of the subtree stored in each node
    """
    curr_node = node

    #for each node from the current node to the root, increment its subtree size by 1
    while curr_node is not None:
        curr_node.update_subtree_size()
        curr_node = curr_node.get_parent()




def tree_policy(root, env: Environment, explore_exploit_const, reward_policy):
    """ Travel down the tree to the ideal node to expand on

    This function loops down the tree until it finds a
    node whose children have not been fully explored, or it
    explores the best child node of the current node.

    Keyword arguments:
    root -- the root node of the tree
    env -- Environment interface between the learning agent and the game
    Return: the ideal node to expand on
    """
    node = root
    count = 0
    first_step = None
    while not node.is_terminal():
        count = count+1
        #if parent is not fully expanded, expand it and return
        if not node.is_expanded():
            new_node = expand_node(node,env)
            if count == 1:
                first_step = new_node.get_prev_action()
            return new_node, first_step
        #Otherwise, look at the parent's best child
        else:
            # Select the best child of the current node to explore
            child = best_child(node, explore_exploit_const, env, reward_policy)[0]
            if count == 1:
                first_step = child.get_prev_action()
            # else, go into the best child
            node = child
            # update the env variable
            env.step(node.get_prev_action())

    # The node is terminal, so return it
    return node, first_step

def best_child(parent, exploration, env: Environment, reward_policy, use_bound = True):
    """ Select and return the best child of the parent node to explore or the action to take

    From the current parent node, we will select the best child node to
    explore and return it. The exploration constant is inputted into this function,
    it balances exploration with exploitation. If the parent node has unexplored
    children, they will automatically be explored first.

    or 

    From the availble actions from this node, we will pick the one that has the most 
    efficient score / visited ratio. Aka the best action to take

    Keyword arguments:
    parent -- the parent node
    exploration -- the exploration-exploitation constant
    use_bound -- whether you are picking the best child to expand (true) or selecting the best action (false)
    Return: the best child to explore in an array with the difference in score between the first and second pick
    """
    max_val = -inf
    bestLs = [None]
    second_best_score = -inf
    for child in parent.get_children():
        # Use the Upper Confidence Bounds for Trees to determine the value for the child or pick the child based on visited
        if(use_bound):
            child_value = reward_policy.upper_confidence_bounds(env, exploration, child.sim_value, child.visited, parent.visited)
        else:
            child_value = reward_policy.select_action(env, child.sim_value, child.visited, parent.visited)
        
        # if there is a tie for best child, randomly pick one
        if (abs(child_value - max_val) < 0.000000001):
            bestLs.append(child)
            second_best_score = child_value
            
        #if it's value is greater than the best so far, it will be our best so far
        elif child_value > max_val:
            second_best_score = max_val
            bestLs = [child]
            max_val = child_value
        #if it's value is greater than the 2nd best, update our 2nd best
        elif child_value > second_best_score:
            second_best_score = child_value
    chosen = random.choice(bestLs)
    if( not use_bound):
        print("best, second", max_val, second_best_score)
    return chosen, abs(max_val - second_best_score) 

def expand_node(parent, env:Environment):
    """
    Expand this node

    Create a random child of this node

    Keyword arguments:
    parent -- the node being expanded
    env -- Environment interface between the learning agent and the game
    Return: a child node to explore
    """
    print(random.getstate())
    # Get possible unexplored actions
    actions = parent.new_actions 
    action = random.choice(actions)

    # Remove that action from the unexplored action list and update parent
    parent.remove_action(action)
    # Step into the state of that child and get its possible actions
    env.step(action)
    new_actions = env.get_valid_actions()
    # Create the child
    new_node = MCTS_node(parent, action, new_actions, env.get_score())

    # Add the child to the parent
    parent.add_child(new_node)
    return new_node

def print_arr(arr):
    str = ""
    for ar in arr:
         str = str+ " "+ar.get_prev_action()
    return str
        


def default_policy(new_node, env,  simulation_length, alpha):
    """
    The default_policy represents a simulated exploration of the tree from
    the passed-in node to a terminal state.

    Self-note: This method doesn't require the nodes to store their depth
    """
    #if node is already terminal, return 0    
    if(env.game_over()):
        #return 0
        return ((new_node.get_score() - new_node.get_parent.get_score()),0)

    scores = [env.get_score()]
    count = 0
    score_states = 0
    # While the game is not over and we have not run out of moves, keep exploring
    while (not env.game_over()) and (not env.victory()) and count < simulation_length:
        count += 1

        #Get the list of valid actions from this state


        # Take a random action from the list of available actions
        actions = env.get_valid_actions()
        env.step(random.choice(actions))
        
        #if there was an increase in the score, add it to the running total
        scores.append(env.get_score())

    discounted_score = 0
    for (i,s) in enumerate(scores):
        if i == 0:
            discounted_score = scores[0]
        else:
            diff = scores[i] - scores[i-1]
            if diff != 0:
                discounted_score += diff*pow(alpha,i)
                score_states += 1
    return (discounted_score,score_states)


def backup(node, delta):
    """
    This function backpropogates the results of the Monte Carlo Simulation back up the tree

    Keyword arguments:
    node -- the child node we simulated from
    delta -- the component of the reward vector associated with the current player at node v
    """
    while node is not None:
        # Increment the number of times the node has
        # been visited and the simulated value of the node
        node.update_visited(1)
        node.update_sim_value(delta)
        # Traverse up the tree
        node = node.get_parent()
"""

def dynamic_sim_len(max_nodes, sim_limit, diff) -> int:
        #
        Given the current simulation depth limit and the difference between 
        the picked and almost picked 'next action' return what the new sim depth and max nodes are.
        
        Keyword arguments:
        max_nodes (int): The max number of nodes to generate before the agent makes a move
        sim_limit (int): The max number of moves to make during a simulation before stopping
        diff (float): The difference between the scores of the best action and the 2nd best action

        Returns: 
            int: The new max number of nodes to generate before the agent makes a move
            int: The new max number of moves to make during a simulation before stopping
        #       
    # if(diff == 0):
            #sim_limit = 100
            #if(max_nodes < 300):
                #max_nodes = max_nodes*2

        if(diff < 0.001):
            if(sim_limit < 1000):
                sim_limit = sim_limit*1.25
            max_nodes = max_nodes+10

        elif(diff > .1):
            if(sim_limit > 12):
                sim_limit =  floor(sim_limit/1.25)
            
        
        return max_nodes, sim_limit
"""
        

def node_explore(agent):
    depth = 0

    cur_node = agent.root

    test_input = "-----"

    chosen_path = agent.node_path

    node_history = agent.node_path

    while test_input != "":
    
        print("\n")

        if(input == ""):
            break

        print("Current Depth:", depth)

        for i in range(0, len(node_history)):
            if depth == 0:
                print(i, "-", node_history[i].get_prev_action())
            else:
                print(i, "-", node_history[i].get_prev_action())

        print("\n")

        test_input = input("Enter the number of the node you wish to explore. Press enter to stop, -1 to go up a layer")

        print("\n")

        if(int(test_input) >= 0 and int(test_input) < len(node_history)):
            depth += 1
            cur_node = node_history[int(test_input)]
        
            print("-------", cur_node.get_prev_action(), "-------")
        
            # print("Sim-value:", cur_node.get_sim_value())
        
            print("Visited:", cur_node.get_visited())
        
            print("Unexplored Children:", cur_node.get_new_actions())
        
            print("Children:")
        
            node_history = cur_node.get_children()
            for i in range(0, len(node_history)):
                print(node_history[i].get_prev_action(), "with value", node_history[i].get_sim_value(), "visited", node_history[i].get_visited())
        elif test_input == "-1":
            depth -= 1
            if depth == 0:
                node_history = agent.node_path
            else:
                cur_node = cur_node.get_parent()
                node_history = cur_node.get_children()

            print("-------", cur_node.get_prev_action(), "-------")
        
            # print("Sim-value:", cur_node.get_sim_value())
        
            print("Visited:", cur_node.get_visited())
        
            print("Unexplored Children:", cur_node.get_new_actions())
        
            print("Children:")

            for i in range(0, len(node_history)):
                was_taken = bool(node_history[i] in chosen_path)                

                print(node_history[i].get_prev_action(), "with value", node_history[i].get_sim_value(), "visited", node_history[i].get_visited(), "was_chosen?", was_taken)
