"""
An implementation of the UCT algorithm for text-based games
"""
from math import floor, inf
from platform import architecture
import random
from xmlrpc.client import Boolean
from environment import *
from mcts_node import Node
from mcts_reward import *
ACTION_BOUND = .001
SIM_SCALE = .15


def take_action(queue_list, env: Environment, explore_exploit_const, simulation_length, reward_policy, timer, procs_finished, lock):
    """
    Take action continuously expands and explores the tree from the root passed in.
    It runs until a timer from the main thread tells it to stop, or until it calculates 10,000 nodes.

    Keyword arguments:
    queue_list: shared queue that stores the score dictionary, count dictionary, and root
    env: game environment
    explore_exploit_const: constant that determines balance between exploring new path versus taking old path
    simulation_length: length of the simulation to run from each new node
    reward_policy: determines which reward policy to use
    timer: shared timer will set to false when the time is up, and this function will return
    procs_finished: shared integer that holds the number of processes that have returned from take_action
    lock: locks procs_finished so only one process can edit at a time
    """

    #get the dictionaries off of the multiprocessing queue
    score_dict = queue_list.get()
    count_dict = queue_list.get()
    root = queue_list.get()
    curr_state = env.get_state()
    count = 0
    action = None
    sim_length_scale = 1
    num_children = len(env.get_valid_actions())

    #if the children of the root have been thoroughly explored and there are no 
    #obvious score differences, adjust the simulation length before exploring further down
    #this tree
    if calc_score_dif(root) <=ACTION_BOUND and root.get_prev_action() is not None and root.subtree_size >= num_children*3:
        sim_length_scale = sim_length_scale/num_children

    while(timer.value==0 and count < 10000):

            count = count+1

            #create a new node on the tree and store the action taken to get there
            new_node,action = tree_policy(root, env, explore_exploit_const, reward_policy)

            #update the size of the tree
            update_tree(new_node)

            # Determine the simulated value of the new node
            delta = default_policy(new_node, env, sim_length_scale* simulation_length, reward_policy)

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

    #after leaving the action sequence, place the dictionaries back on the shared queue
    queue_list.put(score_dict)
    queue_list.put(count_dict)
    queue_list.put(root)
   
   #increment value of finished processes before returning
    lock.acquire()
    procs_finished.value +=1
    lock.release()
    print("TREE TOTAL RUNS = ",count)
    return 




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
    # Get possible unexplored actions
    actions = parent.new_actions 
    action = random.choice(actions)

    # Remove that action from the unexplored action list and update parent
    actions.remove(action)
    # Step into the state of that child and get its possible actions
    env.step(action)
    new_actions = env.get_valid_actions()
    # Create the child
    new_node = Node(parent, action, new_actions)
    # Add the child to the parent
    parent.add_child(new_node)
    return new_node

def print_arr(arr):
    str = ""
    for ar in arr:
         str = str+ " "+ar.get_prev_action()
    return str
        


def default_policy(new_node, env, sim_length, reward_policy):
    """
    The default_policy represents a simulated exploration of the tree from
    the passed-in node to a terminal state.

    Self-note: This method doesn't require the nodes to store their depth
    """
    #if node is already terminal, return 0    
    if(env.game_over()):
        #return 0
        return env.get_score()

    running_score = env.get_score()
    count = 0
    # While the game is not over and we have not run out of moves, keep exploring
    while (not env.game_over()) and (not env.victory()):
        count += 1
        # if we have reached the limit for exploration
        if(env.get_moves() > sim_length):
            #return the reward received by reaching terminal state
            return running_score

        #Get the list of valid actions from this state
        actions = env.get_valid_actions()

        # Take a random action from the list of available actions
        before = env.get_score()
        env.step(random.choice(actions))
        after = env.get_score()
        
        #if there was an increase in the score, add it to the running total
        if((after-before) > 0):
            running_score += (after-before)/(count+1)

    #return the reward received by reaching terminal state
    return running_score

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
        node.visited += 1
        node.sim_value += delta
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
            #if(max_nodes > 100):
                #max_nodes = floor(max_nodes/2)
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
        
            print("Sim-value:", cur_node.sim_value)
        
            print("Visited:", cur_node.visited)
        
            print("Unexplored Children:", cur_node.new_actions)
        
            print("Children:")
        
            node_history = cur_node.get_children()
            for i in range(0, len(node_history)):
                print(node_history[i].get_prev_action(), "with value", node_history[i].sim_value, "visited", node_history[i].visited)
        elif test_input == "-1":
            depth -= 1
            if depth == 0:
                node_history = agent.node_path
            else:
                cur_node = cur_node.parent
                node_history = cur_node.get_children()

            print("-------", cur_node.get_prev_action(), "-------")
        
            print("Sim-value:", cur_node.sim_value)
        
            print("Visited:", cur_node.visited)
        
            print("Unexplored Children:", cur_node.new_actions)
        
            print("Children:")

            for i in range(0, len(node_history)):
                was_taken = bool(node_history[i] in chosen_path)                

                print(node_history[i].get_prev_action(), "with value", node_history[i].sim_value, "visited", node_history[i].visited, "was_chosen?", was_taken)
