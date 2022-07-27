
from math import sqrt, log, e, inf
import sys
from environment import *
import numpy as np

class Reward:
    """Interface for a Reward"""

    def terminal_node(self, env: Environment) -> int:
        """The case when we start the simulation at a terminal state
        
        Keyword arguments:
            env (Environment): Environment interface between the learning agent and the game
        Returns:
            int: The score for the new node
        """
        raise NotImplementedError

    def simulation_limit(self, env: Environment) -> int:
        """The case when we reach the simulation depth limit
        
        Keyword arguments:
            env (Environment): Environment interface between the learning agent and the game
        Returns:
            int: The score for the new node
        """
        raise NotImplementedError

    def simulation_terminal(self, env: Environment) -> int:
        """The case when we reach a terminal stae in the simulation
        
        Keyword arguments:
            env (Environment): Environment interface between the learning agent and the game
        Returns:
            int: The score for the new node
        """
        raise NotImplementedError
        
    def upper_confidence_bounds(self, env: Environment, exploration, child_sim_value, child_visited, parent_visited) -> int:
        """ This method calculates and returns the upper confidence bounds for a given child node on the tree.

        Args:
            env (Environment): Environment interface between the learning agent and the game
            exploration (float): Exploration-Exploitation constant
            child_sim_value (float): Simulated value for the child node
            child_visited (int): Number of times the child node has been explored
            parent_visited (int): Number of times the parent node has been explored

        Raises:
            NotImplementedError: throw an error if this method is not implemented.

        Returns:
            int: The upper confidence bounds for the child node
        """
        raise NotImplementedError

    def select_action(self, env: Environment, child_sim_value, child_visited, parent_visited) -> int:
        """ This method calculates and returns the average score for a given child node on the tree.

        Args:
            env (Environment): Environment interface between the learning agent and the game
            child_sim_value (float): Simulated value for the child node
            child_visited (int): Number of times the child node has been explored
            parent_visited (int): Number of times the parent node has been explored

        Raises:
            NotImplementedError: throw an error if this method is not implemented.

        Returns:
            int: The average score for the child node
        """
        raise NotImplementedError

class SoftmaxReward(Reward):
    """Softmax reward returns values from 0 to .5 for the state. 
    This implementation assumes that every score between the loss state and the max score
    are possible.
    """

    def terminal_node(self, env: Environment):
        """ The case when we start the simulation at a terminal state
        Keyword arguments:
            env (Environment): Environment interface between the learning agent and the game
        Returns:
            int: The score for the new node
        """
        return 0

    def calculate_child_value(self, env: Environment, child, parent):
        parent_visited = parent.get_visited()
        child_visited = child.get_visited()
        child_sim_reward = child.get_sim_value()

        # TODO: If the child is a terminal state (a loss), return negative infinity
        if child_visited != 0:
            first_term = child_sim_reward/child_visited
            second_term = sqrt((2*log(parent_visited))/child_visited)
            return first_term + self.exploration*second_term
        else:
            return inf * -1
        """
        Args:
            env (Environment): Environment interface between the learning agent and the game
            exploration (float): Exploration-Exploitation constant
            child_sim_value (float): Simulated value for the child node
            child_visited (int): Number of times the child node has been explored
            parent_visited (int): Number of times the parent node has been explored
        
        Raises:
            NotImplementedError: throw an error if this method is not implemented.

        Returns:
            int: The upper confidence bounds for the child node
        """
        if env.get_score() >= np.log(sys.maxsize):
            denom = np.log(sys.maxsize)
        else:
            denom = self.softmax_calc(-10,env.get_max_score())
        if child_sim_value >= np.log(sys.maxsize):
            num = np.log(sys.maxsize)
        else:
            num = child_sim_value
        
        return (e**(num))/(child_visited*denom)+ exploration*sqrt((2*log2(parent_visited))/child_visited)

    def select_action(self, env: Environment, child_sim_value, child_visited, parent_visited):
        """ This method calculates and returns the average score for a given child node on the tree.

        Args:
            env (Environment): Environment interface between the learning agent and the game
            child_sim_value (float): Simulated value for the child node
            child_visited (int): Number of times the child node has been explored
            parent_visited (int): Number of times the parent node has been explored

        Raises:
            NotImplementedError: throw an error if this method is not implemented.

        Returns:
            int: The average score for the child node
        """
        if env.get_max_score() >= np.log(sys.maxsize):
            denom = np.log(sys.maxsize)
        else:
            denom = self.softmax_calc(-10,env.get_max_score())
        if child_sim_value >= np.log(sys.maxsize):
            num = np.log(sys.maxsize)
        else:
            num = child_sim_value
        return (e**(num))/(child_visited*denom)

class Generalized_Softmax_Reward(Reward):
    """Generalized Softmax reward returns values from 0 to 1 for the state. 
    This implementation assumes that every score between the loss state and the max score
    are possible.
    """

    def terminal_node(self, env: Environment):
        """ The case when we start the simulation at a terminal state 

        Keyword arguments:
            env (Environment): Environment interface between the learning agent and the game
        Returns:
            int: The score for the new node
        """
        return 0

    def simulation_limit(self, env: Environment):
        """ The case when we reach the simulation depth limit 
        
        Keyword arguments:
            env (Environment): Environment interface between the learning agent and the game
        Returns:
            int: The score for the new node
        """
        return env.get_score()

    def simulation_terminal(self, env: Environment):
        """ The case when we reach a terminal state in the simulation 
        
        Keyword arguments:
            env (Environment): Environment interface between the learning agent and the game
        Returns:
            int: The score for the new node
        """
        raise env.get_score()+10
   
    def upper_confidence_bounds(self, env: Environment, exploration, child_sim_value, child_visited, parent_visited):
        """ This method calculates and returns the upper confidence bounds for a given child node on the tree.

        Args:
            env (Environment): Environment interface between the learning agent and the game
            exploration (float): Exploration-Exploitation constant
            child_sim_value (float): Simulated value for the child node
            child_visited (int): Number of times the child node has been explored
            parent_visited (int): Number of times the parent node has been explored

        Raises:
            NotImplementedError: throw an error if this method is not implemented.

        Returns:
            int: The upper confidence bounds for the child node
        """
        if env.get_score() >= np.log(sys.maxsize):
            denom = np.log(sys.maxsize)
        else:
            denom = e**(env.get_score())
        if child_sim_value >= np.log(sys.maxsize):
            num = np.log(sys.maxsize)
        else:
            num = child_sim_value
        try:
            return (1/child_visited)*(e**(num-denom)) + exploration*sqrt((2*log2(parent_visited))/child_visited)
        except OverflowError:
            print("max size = ",sys.maxsize," num = ",num," denom = ",denom)

    def select_action(self, env: Environment, child_sim_value, child_visited, parent_visited):
        """ This method calculates and returns the average score for a given child node on the tree.

        Args:
            env (Environment): Environment interface between the learning agent and the game
            child_sim_value (float): Simulated value for the child node
            child_visited (int): Number of times the child node has been explored
            parent_visited (int): Number of times the parent node has been explored

        Raises:
            NotImplementedError: throw an error if this method is not implemented.

        Returns:
            int: The average score for the child node
        """
        if env.get_score() >= np.log(sys.maxsize):
            denom = np.log(sys.maxsize)
        else:
            denom = e**(env.get_score())
        if child_sim_value >= np.log(sys.maxsize):
            num = np.log(sys.maxsize)
        else:
            num = child_sim_value
        try:
            return (1/child_visited)*(e**(num-denom))
        except OverflowError:
            print("max size = ",sys.maxsize," num = ",num," denom = ",denom)

class AdditiveReward(Reward):
    """ This Reward Policy returns values between 0 and 1 
    for the state inputted state.
    """

    def terminal_node(self, env: Environment):
        """The case when we start the simulation at a terminal state, return 0.
        
        Keyword arguments:
            env (Environment): Environment interface between the learning agent and the game
        Returns:
            int: The score for the new node
        """
        return 0

    def simulation_limit(self, env: Environment):
        """The case when we reach the simulation depth limit
        
        Keyword arguments:
            env (Environment): Environment interface between the learning agent and the game
        Returns:
            int: The score for the new node
        """
        return env.get_score()

    def simulation_terminal(self, env: Environment):
        """The case when we reach a terminal stae in the simulation. 
        Add 10 to the score so it is non-negative.
        
        Keyword arguments:
            env (Environment): Environment interface between the learning agent and the game
        Returns:
            int: The score for the new node
        """
        return env.get_score()+10

    def upper_confidence_bounds(self, env: Environment, exploration, child_sim_value, child_visited, parent_visited):
        """ This method calculates and returns the upper confidence bounds for a given child node on the tree.

        Args:
            env (Environment): Environment interface between the learning agent and the game
            exploration (float): Exploration-Exploitation constant
            child_sim_value (float): Simulated value for the child node
            child_visited (int): Number of times the child node has been explored
            parent_visited (int): Number of times the parent node has been explored

        Raises:
            NotImplementedError: throw an error if this method is not implemented.

        Returns:
            int: The upper confidence bounds for the child node
        """
        score = env.get_score()
        if score == 0:
            score = 1
        #print(child_sim_value/(child_visited*score),  exploration*sqrt((2*log2(parent_visited))/child_visited))
        return child_sim_value/(child_visited*score) + 1.75*exploration*sqrt((2*log2(parent_visited))/child_visited)

#         Returns:
#             int: The average score for the child node
#         """
#         return (child_sim_value/(child_visited*env.get_max_score()))
