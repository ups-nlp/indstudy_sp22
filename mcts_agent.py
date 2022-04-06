"""
An implementation of the UCT algorithm for text-based games
"""
from math import floor, inf
import random
from environment import *
from mcts_node import Node, MCTS_node
from mcts_reward import *


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
    # How do you go back up the tree to explore other paths
    # when the best path has progressed past the max_depth?
    #while env.get_moves() < max_depth:
    while not node.is_terminal():
        #if parent is not full expanded, expand it and return
        if not node.is_expanded():
            return expand_node(node, env)
        #Otherwise, look at the parent's best child
        else:
            # Select the best child of the current node to explore
            child = best_child(node, explore_exploit_const, env, reward_policy)[0]
            # else, go into the best child
            node = child
            # update the env variable
            env.step(node.get_prev_action())

    # The node is terminal, so return it
    return node

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
    if( not use_bound):
        print("best, second", max_val, second_best_score)
    return chosen, abs(max_val - second_best_score) ## Worry about if only 1 node possible infinity?

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

    #print(len(actions), rand_index)
    action = random.choice(actions)

    # Remove that action from the unexplored action list and update parent
    actions.remove(action)

    # Step into the state of that child and get its possible actions
    env.step(action)
    new_actions = env.get_valid_actions()

    # Create the child
    new_node = MCTS_node(parent, action, new_actions)

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

        # NOTE: It would spend so much time simulating in the default policy, that we'd only
        # get 5-15 simulations done
        if(count > sim_length):
            #return the reward received by reaching terminal state
            #return reward_policy.simulation_limit(env)
            return running_score

        #Get the list of valid actions from this state
        actions = env.get_valid_actions()

        # Take a random action from the list of available actions
        before = env.get_score()
        env.step(random.choice(actions))
        after = env.get_score()
        
        #if there was an increase in the score, add it to the running total
        if((after-before) > 0):
            running_score += (after-before)/count

    #return the reward received by reaching terminal state
    #return reward_policy.simulation_terminal(env)
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
        node.update_visited(1)
        node.update_sim_value(delta)
        # Traverse up the tree
        node = node.get_parent()

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
        if(diff < 0.001):
            if(sim_limit < 1000):
                sim_limit = sim_limit*1.25
            max_nodes = max_nodes+10

        elif(diff > .1):
            if(sim_limit > 12):
                sim_limit =  floor(sim_limit/1.25)
            
        
        return max_nodes, sim_limit

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
        
            print("Sim-value:", cur_node.get_sim_value())
        
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
        
            print("Sim-value:", cur_node.get_sim_value())
        
            print("Visited:", cur_node.get_visited())
        
            print("Unexplored Children:", cur_node.get_new_actions())
        
            print("Children:")

            for i in range(0, len(node_history)):
                was_taken = bool(node_history[i] in chosen_path)                

                print(node_history[i].get_prev_action(), "with value", node_history[i].get_sim_value(), "visited", node_history[i].get_visited(), "was_chosen?", was_taken)
