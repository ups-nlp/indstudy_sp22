"""
An implementation of the UCT algorithm for text-based games
"""
from math import floor, inf
import config
from config import random
from environment import *
from mcts_node import MCTS_node
from mcts_reward import *

def tree_policy(root, env: Environment, reward_policy):
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
    while not node.is_terminal():
        #if parent is not fully expanded, expand it and return
        if not node.is_expanded():
            return expand_node(node, env)
        #Otherwise, look at the parent's best child
        else:
            # Select the best child of the current node to explore
            child = best_child(node, env, reward_policy)
            node = child            
            env.step(node.get_prev_action())

    # The node is terminal, so return it
    return node

def best_child(parent, env: Environment, reward_policy):
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

    # check the pre-condition
    if(not parent.is_expanded()):    
        print("ALERT: best_child() called on parent node that has not been fully expanded")
        print("Num children", parent.get_max_children())


    max_val = -inf
    bestLs = [None]
    tolerance = 0.000000001

    for child in parent.get_children():

        child_value = reward_policy.calculate_child_value(env, child, parent)
        
        # if there is a tie for best child, randomly pick one        
        if (abs(child_value - max_val) < tolerance):
            bestLs.append(child)
            
        #if it's value is greater than the best so far, it will be our best so far
        elif child_value > max_val:
            bestLs = [child]
            max_val = child_value

    chosen = random.choice(bestLs)

    # Colin is stil computing the 2nd best but he's not using it programmatically (but wants to keep)
    # Anna is not using the 2nd best anymore
    return chosen

def expand_node(parent, env):
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
    action = random.choice(actions)

    # Remove that action from the unexplored action list and update parent
    parent.remove_action(action)

    # Step into the state of that child and get its possible actions
    env.step(action)
    new_actions = env.get_valid_actions()

    # Create the child
    new_node = MCTS_node(parent, action, new_actions)

    # Add the child to the parent
    parent.add_child(new_node)

    return new_node
  
def default_policy(new_node, env):
    """
    The default_policy represents a simulated exploration of the tree from
    the passed-in node to a terminal state.

    Self-note: This method doesn't require the nodes to store their depth
    """
    #if node is already terminal, return 0    
    if(env.game_over()):
        return env.get_score()
    
    count = 0

    # While the game is not over and we have not run out of moves, keep exploring
    while (not env.game_over()) and (not env.victory()):        

        #Get the list of valid actions from this state
        actions = env.get_valid_actions()

        # Take a random action from the list of available actions        
        env.step(random.choice(actions))        

        count += 1    
           

    #if config.VERBOSITY > 0:
    #    print('\t[DEFAULT POLICY] Number of iterations until reached terminal node: ', count)

    return env.get_score()



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


