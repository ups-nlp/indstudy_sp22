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
    path = [root]
    while not node.is_terminal():

        #if parent is not fully expanded, expand it and return
        if not node.is_expanded():
            new_node = expand_node(node, env, transposition_table)
            path.append(new_node)
            return new_node, path

        #Otherwise, look at the parent's best child
        else:
            # Select the best child of the current node to explore
            child = best_child(node, explore_exploit_const, env, reward_policy, True)

            # add child to the path
            node = child
            path.append(node)
            # update the env variable
            env.step(node.get_prev_action())

    # The node is terminal, so return it
    return node, path


def best_child(parent, exploration, env: Environment, reward_policy, use_bound = True):
    """ Select and return the best child of the parent node to explore or the action to take

    pre: parent has been fully expanded

    For each child node (v) of the parent, we compute:
    Q(v)/N(v) + e * sqrt(2ln[N(parent)]/N(v))

    where 
    
    Q(v)      = The sum of the backed up rewards for the child v
    N(v)      = The total visit count for the child v
    e         = A user-chosen constant that trades off btw. exploration and exploitation
    N(parent) = The total visit count for the parent 

    The child node with the highest value is returned. 

    Keyword arguments:
    parent -- the parent node
    exploration -- the exploration-exploitation constant
    use_bound -- whether you are picking the best child to expand (true) or selecting the best action (false)
    Return: the best child to explore in an array with the difference in score between the first and second pick
    """
    fullyExpanded = True

    # check the precondition
    if not parent.is_expanded():
        print("ALERT: best_child() called on parent node that has not been fully expanded")
        print("ALERT: Num children", parent.get_max_children())
        fullyExpanded = False

    if len(parent.get_children()) == 0:
        exit("ALERT: This parent has 0 expanded children")
        
    max_val = -inf
    bestLs = [None]
    tolerance = 0.000000001

    for child in parent.get_children():
        
        child_value = reward_policy.calculate_child_value(env, child, parent)
        
        # if there is a tie for best child, randomly pick one        
        if abs(child_value - max_val) < tolerance:
            bestLs.append(child)            
            
        #if it's value is greater than the best so far, it will be our best so far
        elif child_value > max_val:
            bestLs = [child]
            max_val = child_value        

    chosen = random.choice(bestLs)        
    return chosen


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
    actions.remove(action)

    # Step into the state of that child and get its possible actions
    env.step(action)    
    new_actions = env.get_valid_actions()

    # Create the child
    # new_node = Node(parent, action, new_actions)
    state = get_world_state_hash(env.get_player_location(), new_actions)
    new_node = Transposition_Node(state, parent, action, new_actions, transposition_table)

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
   
    # While the game is not over and we have not run out of moves, keep exploring
    while (not env.game_over()) and (not env.victory()):

        #Get the list of valid actions from this state
        actions = env.get_valid_actions()

        # Take a random action from the list of available actions
        action = random.choice(actions)
        env.step(action)

    return env.get_score()



# TODO: I need to take some time to think about this. I think this is one of the
# important parts of Colin's algorithm since. I'm going to pause here and go back
# to the basic algorithm and implement discount rewarding for the default policy
# I think for our EMNLP paper, I'm going to focus on comparing Anna's parallelized
# version to the basic version with some additional discussion of modifications of the
# simulation length. 
#
# In the future, I need to pick up here. Re-read Colin's related work and then 
# carefully go through this method and understand how the path plays a role
def backup(path, delta):
    """
    This function backpropogates the results of the Monte Carlo Simulation back up the tree

    Keyword arguments:
    node -- the child node we simulated from
    delta -- the component of the reward vector associated with the current player at node v
    """
    # get the length of the path
    max_size = len(path)
    # Keep a list of the states we have updated
    updated_states = []
    for index in reversed(range(max_size)):
        # Increment the number of times the node has
        # been visited and the simulated value of the node
        node = path[index]
        if(not updated_states.__contains__(node.get_state())):
            # Increment the count of the state
            node.update_visited(1)

            # use decaying rewards?
            # alpha = delta/2**(max_size - (index+1))
            alpha = delta
            # Update the score of the state
            node.update_sim_value(alpha)

            # Add the state to our list of updated states
            updated_states.append(node.get_state())


