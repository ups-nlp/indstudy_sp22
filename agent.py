"""
Agents for playing text-based games
"""

import random
from jericho import FrotzEnv


class Agent:
    """Interface for an Agent"""

    def take_action(self, env: FrotzEnv, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        raise NotImplementedError


class RandomAgent(Agent):
    """Agent randomly selects an action from list of valid actions"""

    def take_action(self, env: FrotzEnv, history: list) -> str:
        """Takes in the history and returns the next action to take"""

        valid_actions = env.get_valid_actions()
        return random.choice(valid_actions)

class MonteAgent(Agent):
    node_path = []

    def __init__(self, env: FrotzEnv, game_file:str, num_steps: int):

        action_values = {}
        action_counter = {}
        self.take_time = True
        #create threads
        # for each thread, give them a game to play with:
        envThing = FrotzEnv(game_file)

    def take_action(self, env: FrotzEnv, history: list) -> str:
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

        #spin threads to begin their process
        while((seconds_elapsed < time_limit or count <= minimum)):
           #wait here
           x = 1
        self.take_time = False
        
        



class HumanAgent(Agent):
    """Allows a human player"""

    def take_action(self, env: FrotzEnv, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        print("Action: ")
        return input()
