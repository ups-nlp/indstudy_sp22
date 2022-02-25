"""
An interface for a game environment
"""

from copy import deepcopy
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

    def reset(self):
        """Reset the environment"""
        raise NotImplementedError

    def get_player_location(self):
        """Returns the player's location"""
        raise NotImplementedError

    def get_state(self):
        """Returns the internal game state"""
        raise NotImplementedError    

    def set_state(self, state):
        """Sets the internal game state to the specified state"""
        raise NotImplementedError  

    def get_world_state_hash(self):
        """Get a hash of the current state of the game"""
        raise NotImplementedError
    


class JerichoEnvironment(Environment):    
    """A wrapper around the FrotzEnvironment"""

    def __init__(self, path: str):
        self.env = FrotzEnv(path)
    
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

    def reset(self):
        """Reset the environment"""
        return self.env.reset()

    def get_player_location(self):
        """Returns the player's location"""
        return self.env.get_player_location()

    def get_state(self):
        """Returns the internal game state"""
        return self.env.get_state()
    

    def set_state(self, state):
        """Sets the internal game state to the specified state"""
        return self.env.set_state(state)

    def get_world_state_hash(self):
        """Get a hash of the current state of the game"""
        return self.env.get_world_state_hash()



class ChamberEnvironment(Environment):    
    """Implementes a simple game with only 1 location
    
    The user is in a chamber with a door. To win, the user must open the door.
    All other actions have no effect. The reward is 1 divided by the number of moves
    it took to open the door. If the user has not won after the maximum number of
    (valid) moves allowed, the game automatically ends with a loss.
    """

    # class-level variables
    max_num_moves = 4
    initial_obs = "This is a chamber. You can see a (closed) door here."
    losing_obs = "You failed to open the door. You lose."
    winning_obs = "You open the door. You have won!"
    location = "chamber"
    actions = ["n", "e", "s", "w", "open", "fight", "look"]

    def __init__(self, path: str):
        self.moves = []        
        self.is_game_over = False
        self.is_victory = False
        self.last_obs = None    # Final observation when the game ends 
        self.reward = 0         # Final reward when game ends
        
    
    def step(self, action: str):
        """Takes an action and returns the next state, reward, and termination"""

        # The game is over
        if self.is_game_over:
            return (self.last_obs, self.reward, self.is_game_over, {'moves' : len(self.moves), 'score' : self.reward})
     
        # Game is not over...process the player's action
        response = self.__get_response_string(action) 

        # An invalid action (nothing about the state changes)
        if action not in ChamberEnvironment.actions:            
            return (response, 0, False, {'moves': len(self.moves), 'score' : 0})

        # A valid action: record the action
        self.moves.append(action)

        # Did they just win?!
        if response == ChamberEnvironment.winning_obs:
            self.is_game_over = True
            self.is_victory = True
            self.last_obs = ChamberEnvironment.winning_obs
            self.reward = 1.0 / len(self.moves)            
            return (self.last_obs, self.reward, self.is_game_over, {'moves': len(self.moves), 'score' : self.reward}) 

        # Did they just lose (i.e. this is their last move and they didn't win)
        if len(self.moves) == ChamberEnvironment.max_num_moves:                        
            self.is_game_over = True
            self.is_victory = False
            self.last_obs = ChamberEnvironment.losing_obs
            self.reward = -1            
            return (self.last_obs, self.reward, self.is_game_over, {'moves': len(self.moves), 'score' : self.reward})
                
        # Otherwise, it's just a plain ole' turn of the game
        return (response, 0, False, {'moves': len(self.moves), 'score' : 0})

        

    def get_valid_actions(self):
        """Attempts to generate a set of unique valid actions from the current game state"""
        return deepcopy(ChamberEnvironment.actions)

    def game_over(self):
        """Returns true if the game is over and the player has lost"""
        return self.is_game_over and not self.is_victory

    def get_score(self):
        """Returns the integer current game score"""
        
        # Game is over, it was a win
        if self.is_game_over:            
            return self.reward
        
        # Game is not yet over
        else:            
            return 0
        
    def victory(self):
        """Returns true if game is over and the player has won"""
        return self.is_game_over and self.is_victory

    def get_moves(self):
        """Returns the integer number of moves taken by the player in the current episode"""
        return len(self.moves)

    def get_max_score(self):
        """Returns the integer maximum possible score for the game"""
        return 1

    def reset(self):
        """Reset the environment"""
        self.moves = []        
        self.is_game_over = False
        self.is_victory = False
        self.last_obs = None
        self.reward = 0         
        return (ChamberEnvironment.initial_obs, {'moves': 0, 'score': 0})

    def get_player_location(self):
        """Returns the player's location"""
        return ChamberEnvironment.location

    def get_state(self):
        """Returns the internal game state"""
        state = {'moves' : self.moves, 'game_over': self.is_game_over, 'victory' : self.is_victory, 'last_obs' : self.last_obs, 'reward': self.reward}        
        return state
    

    def set_state(self, state):
        """Sets the internal game state to the specified state"""
        if 'moves' not in state or 'game_over' not in state or 'victory' not in state or 'last_obs' not in state or 'reward' not in state:
            print('Invalid state passed to method')
            raise ValueError

        if not isinstance(state['moves'], list):
            print('Moves must be a list')
            raise ValueError

        self.moves = state['moves']
        self.is_game_over = state['game_over']
        self.is_victory = state['victory']
        self.last_obs = state['last_obs']
        self.reward = state['reward']
        
    def get_world_state_hash(self):
        """Get a hash of the current state of the game"""
        return 1
    
    def __get_response_string(self, action: str):
        if action == "n" or action == "e" or action == "s" or action == "w":
            return "You can't go that way"
        elif action == "open":
            return ChamberEnvironment.winning_obs
        elif action == "fight":
            return "Violence isn't the answer to this one"
        elif action == "look":
            return self.initial_obs
        else:
            return "That's not a command I recognize."
