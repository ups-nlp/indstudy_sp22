
from math import sqrt, log, e
import sys
from environment import *
import numpy as np

# TODO: Really need to make this a true abstract class using the abc module

class Reward:
    """Abstract reward class"""

    def __init__(self, exploration_constant):
        self.exploration = exploration_constant        

    def calculate_child_value(self, env: Environment, child, parent):
        """ Calculates the child's reward value according to a particular algorithmic strategy
        
            Args: 
            env (Environment): Environment interface between the learning agent and the game
            child (Node) : A child node
            parent (Node) : A parent node
        """
        raise NotImplementedError


class BaselineReward(Reward):
    """Implements the standard upper-confidence bound for trees equation
    
        Browne et al., A survey of monte carlo tree search algorithms. IEEE Trans. on 
        Compt'l Intelligence and AI in Games, vol. 4, no. 10, March 2012.

        Algorithm 2 > BestChild() function
    """

    def __init__(self, exploration_constant):
        super.__init__(exploration_constant)

    def calculate_child_Value(self, env: Environment, child, parent):
        parent_visited = parent.get_visited()
        child_visited = child.get_visited()
        child_sim_reward = child.get_sim_value()

        if child_visited == 0:
            raise ZeroDivisionError("Child has been visited 0 times")
        
        return (child_sim_reward/child_visited) + self.exploration*sqrt((2*log(parent_visited))/child_visited)


class NormalizedReward(Reward):
    """Returns the normalized simulated reward value for the child
    """

    def __init__(self, exploration_constant):
        super.__init__(exploration_constant)

    def calculate_child_value(self, env: Environment, child, parent):
        child_rewards = [child.get_sim_value() for child in parent.get_children()]
        z = sum(child_rewards) # normalizing constant
        
        if z == 0:
            raise ZeroDivisionError("Sum of children's simulated rewards is 0")

        return child.get_sim_value() / z



# class SoftmaxReward(Reward):
#     """Softmax reward returns values from 0 to .5 for the state. 
#     This implementation assumes that every score between the loss state and the max score
#     are possible.
#     """
        
#     def upper_confidence_bounds(self, env: Environment, exploration, child_sim_value, child_visited, parent_visited):
#         """ This method calculates and returns the upper confidence bounds for a given child node on the tree.

#         Args:
#             env (Environment): Environment interface between the learning agent and the game
#             exploration (float): Exploration-Exploitation constant
#             child_sim_value (float): Simulated value for the child node
#             child_visited (int): Number of times the child node has been explored
#             parent_visited (int): Number of times the parent node has been explored

#         Raises:
#             NotImplementedError: throw an error if this method is not implemented.

#         Returns:
#             int: The upper confidence bounds for the child node
#         """
#         if env.get_score() >= np.log(sys.maxsize):
#             denom = np.log(sys.maxsize)
#         else:
#             denom = self.softmax_calc(-10,env.get_max_score())
#         if child_sim_value >= np.log(sys.maxsize):
#             num = np.log(sys.maxsize)
#         else:
#             num = child_sim_value
        
#         return (e**(num))/(child_visited*denom)+ exploration*sqrt((2*log(parent_visited))/child_visited)

#     def select_action(self, env: Environment, child_sim_value, child_visited, parent_visited):
#         """ This method calculates and returns the average score for a given child node on the tree.

#         Args:
#             env (Environment): Environment interface between the learning agent and the game
#             child_sim_value (float): Simulated value for the child node
#             child_visited (int): Number of times the child node has been explored
#             parent_visited (int): Number of times the parent node has been explored

#         Raises:
#             NotImplementedError: throw an error if this method is not implemented.

#         Returns:
#             int: The average score for the child node
#         """
#         if env.get_max_score() >= np.log(sys.maxsize):
#             denom = np.log(sys.maxsize)
#         else:
#             denom = self.softmax_calc(-10,env.get_max_score())
#         if child_sim_value >= np.log(sys.maxsize):
#             num = np.log(sys.maxsize)
#         else:
#             num = child_sim_value
#         return (e**(num))/(child_visited*denom)

# class Generalized_Softmax_Reward(Reward):
#     """Generalized Softmax reward returns values from 0 to 1 for the state. 
#     This implementation assumes that every score between the loss state and the max score
#     are possible.
#     """
   
#     def upper_confidence_bounds(self, env: Environment, exploration, child_sim_value, child_visited, parent_visited):
#         """ This method calculates and returns the upper confidence bounds for a given child node on the tree.

#         Args:
#             env (Environment): Environment interface between the learning agent and the game
#             exploration (float): Exploration-Exploitation constant
#             child_sim_value (float): Simulated value for the child node
#             child_visited (int): Number of times the child node has been explored
#             parent_visited (int): Number of times the parent node has been explored

#         Raises:
#             NotImplementedError: throw an error if this method is not implemented.

#         Returns:
#             int: The upper confidence bounds for the child node
#         """
#         if env.get_score() >= np.log(sys.maxsize):
#             denom = np.log(sys.maxsize)
#         else:
#             denom = e**(env.get_score())
#         if child_sim_value >= np.log(sys.maxsize):
#             num = np.log(sys.maxsize)
#         else:
#             num = child_sim_value
#         try:
#             return (1/child_visited)*(e**(num-denom)) + exploration*sqrt((2*log2(parent_visited))/child_visited)
#         except OverflowError:
#             print("max size = ",sys.maxsize," num = ",num," denom = ",denom)

#     def select_action(self, env: Environment, child_sim_value, child_visited, parent_visited):
#         """ This method calculates and returns the average score for a given child node on the tree.

#         Args:
#             env (Environment): Environment interface between the learning agent and the game
#             child_sim_value (float): Simulated value for the child node
#             child_visited (int): Number of times the child node has been explored
#             parent_visited (int): Number of times the parent node has been explored

#         Raises:
#             NotImplementedError: throw an error if this method is not implemented.

#         Returns:
#             int: The average score for the child node
#         """
#         if env.get_score() >= np.log(sys.maxsize):
#             denom = np.log(sys.maxsize)
#         else:
#             denom = e**(env.get_score())
#         if child_sim_value >= np.log(sys.maxsize):
#             num = np.log(sys.maxsize)
#         else:
#             num = child_sim_value
#         try:
#             return (1/child_visited)*(e**(num-denom))
#         except OverflowError:
#             print("max size = ",sys.maxsize," num = ",num," denom = ",denom)

# class AdditiveReward(Reward):
#     """ This Reward Policy returns values between 0 and 1 
#     for the state inputted state.
#     """

#     def upper_confidence_bounds(self, env: Environment, exploration, child_sim_value, child_visited, parent_visited):
#         """ This method calculates and returns the upper confidence bounds for a given child node on the tree.

#         Args:
#             env (Environment): Environment interface between the learning agent and the game
#             exploration (float): Exploration-Exploitation constant
#             child_sim_value (float): Simulated value for the child node
#             child_visited (int): Number of times the child node has been explored
#             parent_visited (int): Number of times the parent node has been explored

#         Raises:
#             NotImplementedError: throw an error if this method is not implemented.

#         Returns:
#             int: The upper confidence bounds for the child node
#         """
#         score = env.get_score()
#         if score == 0:
#             score = 1
#         #print(child_sim_value/(child_visited*score),  exploration*sqrt((2*log2(parent_visited))/child_visited))
#         return child_sim_value/(child_visited*score) + 1.75*exploration*sqrt((2*log2(parent_visited))/child_visited)

#     def select_action(self, env: Environment, child_sim_value, child_visited, parent_visited):
#         """ This method calculates and returns the average score for a given child node on the tree.

#         Args:
#             env (Environment): Environment interface between the learning agent and the game
#             child_sim_value (float): Simulated value for the child node
#             child_visited (int): Number of times the child node has been explored
#             parent_visited (int): Number of times the parent node has been explored

#         Raises:
#             NotImplementedError: throw an error if this method is not implemented.

#         Returns:
#             int: The average score for the child node
#         """
#         score = env.get_score()
#         if(score == 0):
#             score = 1
#         return child_sim_value/(child_visited*score)

# class DynamicReward(Reward):
#     """Dynamic Reward  scales the reward returned in a simulation by the length of the simulation,
#         so a reward reached earlier in the game will have a higher score than the same state
#          reached later."""

        
#     def upper_confidence_bounds(self, env: Environment, exploration, child_sim_value, child_visited, parent_visited) -> int:
#         """ This method calculates and returns the upper confidence bounds for a given child node on the tree.

#         Args:
#            env (Environment): Environment interface between the learning agent and the game
#             exploration (float): Exploration-Exploitation constant
#             child_sim_value (float): Simulated value for the child node
#             child_visited (int): Number of times the child node has been explored
#             parent_visited (int): Number of times the parent node has been explored

#         Raises:
#             NotImplementedError: throw an error if this method is not implemented.

#         Returns:
#             int: The upper confidence bounds for the child node
#         """
#         return child_sim_value/(child_visited*env.get_max_score()) + exploration*sqrt((2*log2(parent_visited))/child_visited)

#     def select_action(self, env: Environment, child_sim_value, child_visited, parent_visited) -> int:
#         """ This method calculates and returns the average score for a given child node on the tree.

#         Args:
#             env (Environment): Environment interface between the learning agent and the game
#             child_sim_value (float): Simulated value for the child node
#             child_visited (int): Number of times the child node has been explored
#             parent_visited (int): Number of times the parent node has been explored

#         Raises:
#             NotImplementedError: throw an error if this method is not implemented.

#         Returns:
#             int: The average score for the child node
#         """
#         return (child_sim_value/(child_visited*env.get_max_score()))