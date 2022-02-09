"""
An interface for a game environment
"""

from asyncio.windows_events import NULL
from jericho import FrotzEnv

class Environment:
    """Interface for an Environment"""

    def step(self, action: str):
        """Takes an action and returns the next state, reward, and termination"""
        raise NotImplementedError

    def get_valid_actions(self):
        """Attempts to generate a set of unique valid actions from the current game state"""
        raise NotImplementedError

    def game_over(self):
        """Returns true if the game is over and the player has lost"""
        raise NotImplementedError

    def get_score(self):
        """Returns the integer current game score"""
        raise NotImplementedError

    def victory(self):
        """Returns true if game is over and the player has won"""
        raise NotImplementedError

    def get_moves(self):
        """Returns the integer number of moves taken by the player in the current episode"""
        raise NotImplementedError

    def get_max_score(self):
        """Returns the integer maximum possible score for the game"""
        raise NotImplementedError
    


class JerichoEnvironment(Environment):
    env = None

    def __init__(self, path: str):
        env = FrotzEnv(path)
    
    def step(self, action: str):
        """Takes an action and returns the next state, reward, and termination"""
        return self.env.step(action)

    def get_valid_actions(self):
        """Attempts to generate a set of unique valid actions from the current game state"""
        return self.env.get_valid_actions()

    def game_over(self):
        """Returns true if the game is over and the player has lost"""
        return self.env.game_over()

    def get_score(self):
        """Returns the integer current game score"""
        return self.env.get_score()

    def victory(self):
        """Returns true if game is over and the player has won"""
        return self.env.victory()

    def get_moves(self):
        """Returns the integer number of moves taken by the player in the current episode"""
        return self.env.get_moves()

    def get_max_score(self):
        """Returns the integer maximum possible score for the game"""
        return self.env.get_max_score()

