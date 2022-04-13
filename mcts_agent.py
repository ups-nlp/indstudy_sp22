"""
An implementation of the UCT algorithm for text-based games
"""
from math import floor, inf
import random
from environment import *
from mcts_node import Node
from transposition_table import Transposition_Node, get_world_state_hash
from mcts_reward import *
import config


def tree_policy(root, env: Environment, explore_exploit_const, reward_policy, transposition_table):
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
    history = {root.get_state()}
    path = [root] #
    #print("\nstarting at node:", node.get_state().state)
    count = 0
    while not node.is_terminal():
        count += 1
        if(get_world_state_hash(env.get_player_location(), env.get_valid_actions()) != node.get_state().state):
            print("Node-State Mismatch!!! in tree_policy")
            print("env:'", get_world_state_hash(env.get_player_location(), env.get_valid_actions()), "'")
            print("node:'", node.get_state().state, "'")
            sys.exit()
        #!state = get_world_state_hash(env.get_player_location(), env.get_valid_actions())
        #!print("Passed through state: ",state)
        #if parent is not full expanded, expand it and return
        if not node.is_expanded():
            #print("Expand Node")
            new_node = expand_node(node, env, transposition_table)
            #print("exploring node:", new_node.get_state().state)
            #print("with parent node:", new_node.get_parent().get_state().state)
            #print("env:'", get_world_state_hash(env.get_player_location(), env.get_valid_actions()), "'")
            path.append(new_node) #
            return new_node, path

        #Otherwise, look at the parent's best child
        else:
            # Select the best child of the current node to explore
            child, _ = best_child(node, explore_exploit_const, env, reward_policy, history, True)
            #print("--------")
            #print("node: ", node.get_state())
            #for x in node.get_children():
            #    child_sim_value = x.get_sim_value()
            #    child_visited = x.get_visited()
            #    print("\t", x.get_prev_action(), ", count:", child_visited, ", value:", child_sim_value, "normalized:", reward_policy.upper_confidence_bounds(env, explore_exploit_const, child_sim_value, child_visited, node.get_visited()),"best:", reward_policy.select_action(env, child_sim_value, child_visited, None))
            

            # If we have aleady been to all the child nodes, return the current node and path
            if child is None:
                # We are about to loop, so return the last node
                return node, path
            #print("selected: ", child.get_prev_action())

            # else, go into the best child
            node = child
            history.add(node.get_state())
            path.append(node) #
            # update the env variable
            env.step(node.get_prev_action())
            if(config.VERBOSITY > 1):
                if(count == 1):
                    print(node.get_prev_action())
            # print("Entered child: ", node.get_prev_action(), ", env: ", env.get_valid_actions())
            
            #print("entering node:", node.get_state().state, ", with action: ", node.get_prev_action())
            #print("env:'", get_world_state_hash(env.get_player_location(), env.get_valid_actions()), "'")

    # The node is terminal, so return it
    return node, path

def best_child(parent, exploration, env: Environment, reward_policy, history, use_bound = True):
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
    valid_children = []


    if(get_world_state_hash(env.get_player_location(), env.get_valid_actions()) != parent.get_state().state):
        print("Node-State Mismatch!!! in best_child")
        if(use_bound):
            print("best child to expand")
        else:
            print("selecting the best action")
        print(get_world_state_hash(env.get_player_location(), env.get_valid_actions()))
        print(parent.get_state().state)
        sys.exit()
    


    # first we will accrue a list of children nodes with states we have not already been to
    if use_bound:
        for child in parent.get_children():
            # if we have already visited the state of the child, it is not a valid choice
            if(child.get_state() not in history):
                valid_children.append(child)
        # if we have been to all the children's states, they are all valid options
        if len(valid_children) == 0:
            #if we have already been to all the child states, return None
            return None, None
    else:
        valid_children = parent.get_children()
    # calculate the scores for each child and pick the best one
    for child in valid_children:
        
        # Use the Upper Confidence Bounds for Trees to determine the value for the child or pick the child based on visited
        if(use_bound):
            child_value = reward_policy.upper_confidence_bounds(env, exploration, child.get_sim_value(), child.get_visited(), parent.get_visited())
        else:
            child_value = reward_policy.select_action(env, child.get_sim_value(), child.get_visited(), parent.get_visited())
        
        #print("child_value", child_value)
        # if there is a tie for best child, randomly pick one
        # if(child_value == max_val) with floats
        if (abs(child_value - max_val) < 0.000000001):
            
            #print("reoccuring best", child_value)
            #print("next best", child_value)
            bestLs.append(child)
            second_best_score = child_value
            
        #if it's value is greater than the best so far, it will be our best so far
        elif child_value > max_val:
            #print("new best", child_value)
            #print("next best", max_val)
            second_best_score = max_val
            bestLs = [child]
            max_val = child_value
        #if it's value is greater than the 2nd best, update our 2nd best
        elif child_value > second_best_score:
            #print("best", bestLs[0])
            #print("new next best", child_value)
            #print("old next best", second_best_score)
            second_best_score = child_value
    chosen = random.choice(bestLs)
    if(config.VERBOSITY > 1):
        if( not use_bound):
            print("best, second", max_val, second_best_score)
    return chosen, abs(max_val - second_best_score) ## Worry about if only 1 node possible infinity?

def expand_node(parent, env, transposition_table):
    """
    Expand this node

    Create a random child of this node

    Keyword arguments:
    parent -- the node being expanded
    env -- Environment interface between the learning agent and the game
    Return: a child node to explore
    """
    # Get possible unexplored actions
    actions = parent.get_new_actions()

    #print(len(actions), rand_index)
    action = random.choice(actions)

    # Remove that action from the unexplored action list and update parent
    p_state = get_world_state_hash(env.get_player_location(), env.get_valid_actions())
    actions.remove(action)

    # Step into the state of that child and get its possible actions
    env.step(action)
    new_actions = env.get_valid_actions()
    
    #print(new_actions)

    # Create the child
    # new_node = Node(parent, action, new_actions)
    # print("Make new node")

    state = get_world_state_hash(env.get_player_location(), env.get_valid_actions())
    new_node = Transposition_Node(state, parent, action, new_actions, transposition_table)

    if(p_state != parent.get_state().state):
        print("Error!!!!!!! ---------------------------")
        print("\tCreating node of state: ",state, "\n\twith parent: ", p_state,"\n\tand node: ", new_node.toString())
        sys.exit()
    # print("Made new node")
    # Add the child to the parent
    parent.add_child(new_node)

    return new_node


    # # if no new nodes were created, we are at a terminal state
    # if new_node is None:
    #     # set the parent to terminal and return the parent
    #     parent.terminal = True
    #     return parent

    # else:
    #     # update the env variable to the new node we are exploring
    #     env.step(new_node.get_prev_action())
    #     # Return a newly created node to-be-explored
    #     return new_node

def default_policy(new_node, env, sim_length, reward_policy):
    """
    The default_policy represents a simulated exploration of the tree from
    the passed-in node to a terminal state.

    Self-note: This method doesn't require the nodes to store their depth
    """
    #if node is already terminal, return 0    
    if(env.game_over()):
        return 0

    running_score = env.get_score()
    count = 0
    # While the game is not over and we have not run out of moves, keep exploring
    while (not env.game_over()) and (not env.victory()):
        count += 1
        # if we have reached the limit for exploration
        if(count > sim_length):
            #return the reward received by reaching terminal state
            #return reward_policy.simulation_limit(env)
            #if(running_score < 0):
                #return 0
            return running_score

        #Get the list of valid actions from this state
        actions = env.get_valid_actions()

        # Take a random action from the list of available actions
        before = env.get_score()
        action = random.choice(actions)
        env.step(action)
        after = env.get_score()
        
        #if there was an increase in the score, add it to the running total
        if((after-before) > 0):
            #print(action, " resulted in an increase of ", (after-before))
            running_score += (after-before)/count

    #return the reward received by reaching terminal state
    #return reward_policy.simulation_terminal(env)
    #if(running_score < 0):
        #return 0
    return running_score

def backup(path, delta):
    """
    This function backpropogates the results of the Monte Carlo Simulation back up the tree

    Keyword arguments:
    node -- the child node we simulated from
    delta -- the component of the reward vector associated with the current player at node v
    """
    max_size = len(path)
    updated_states = []
    for index in reversed(range(max_size)):
        # Increment the number of times the node has
        # been visited and the simulated value of the node
        node = path[index]
        if(not updated_states.__contains__(node.get_state())):
            node.update_visited(1)
            # decaying rewards
            alpha = delta/2**(max_size - (index+1))
            node.update_sim_value(alpha)
            updated_states.append(node.get_state())

def dynamic_sim_len(max_nodes, sim_limit, diff) -> int:
        """Given the current simulation depth limit and the difference between 
        the picked and almost picked 'next action' return what the new sim depth and max nodes are.
        
        Keyword arguments:
        max_nodes (int): The max number of nodes to generate before the agent makes a move
        sim_limit (int): The max number of moves to make during a simulation before stopping
        diff (float): The difference between the scores of the best action and the 2nd best action

        Returns: 
            int: The new max number of nodes to generate before the agent makes a move
            int: The new max number of moves to make during a simulation before stopping
        """        
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
