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
    
    def copy(self):
        """Returns a copy of the environment"""
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

    def copy(self):
        """Returns a copy of the environment"""
        return self.env.copy()



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
        print()
        print()
        print('HERE LEN', len(self.moves))
        print('MAX LEN', ChamberEnvironment.max_num_moves)
        print(len(self.moves)==ChamberEnvironment.max_num_moves)
        print('HERE REWARD', self.reward)      
        print('GAME OVER', self.is_game_over)
        print('VICTORY', self.is_victory)
        print('ACTION', action)
        print('LAST OBS', self.last_obs)



        # The game is over
        if self.is_game_over:
            print('RETURN GAME OVER')
            return (self.last_obs, self.reward, self.is_game_over, {'moves' : len(self.moves), 'score' : self.reward})
     
        # Game is not over...process the player's action
        response = self.__get_response_string(action) 

        # An invalid action (nothing about the state changes)
        if action not in ChamberEnvironment.actions:            
            print('RETURN INVALID ACTION')
            return (response, 0, False, {'moves': len(self.moves), 'score' : 0})

        # A valid action: record the action
        self.moves.append(action)

        # Did they just win?!
        if response == ChamberEnvironment.winning_obs:
            self.is_game_over = True
            self.is_victory = True
            self.last_obs = ChamberEnvironment.winning_obs
            self.reward = 1.0 / len(self.moves)    
            print('RETURN WIN')             
            return (self.last_obs, self.reward, self.is_game_over, {'moves': len(self.moves), 'score' : self.reward}) 

        # Did they just lose (i.e. this is their last move and they didn't win)
        if len(self.moves) == ChamberEnvironment.max_num_moves:                        
            self.is_game_over = True
            self.is_victory = False
            self.last_obs = ChamberEnvironment.losing_obs
            self.reward = -1    
            print('RETURN LOSS')        
            return (self.last_obs, self.reward, self.is_game_over, {'moves': len(self.moves), 'score' : self.reward})
                
        # Otherwise, it's just a plain ole' turn of the game
        print('RETURN NORMAL')
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

    def copy(self):
        """Returns a copy of the environment"""
        raise NotImplementedError



class Chambers4Environment(Environment):    
    """
    Implements a simple game with 4 chambers:

    |--------------|
    |       |      |
    |  key         |
    |       |      |
    |--  ------  --|
    |       |      |
    |              |
    |       |      |
    |----------D---|
    
    The key is in the top-left chamber. The user begins in the chamber with the door. The user
    must go and get the key and then open the door to win the game. 

    The user gets a reward for picking up the key and for opening the door. 
    If the user has not won after the maximum number of valid moves allowed, the game 
    automatically ends with a loss.
    """

    # class-level variables
    max_num_moves = 10
    initial_obs = "You are in a chamber with a locked door. There is another chamber to your north and west."
    losing_obs = "You failed to open the door. You lose."
    winning_obs = "You open the door. You have won!"
    actions = ["n", "e", "s", "w", "open", "fight", "look", "take key", "drop key"]
    reward_key = 5
    reward_door = 10

    def __init__(self, path: str):
        self.moves = []        
        self.is_game_over = False
        self.is_victory = False
        self.last_obs = None    # Final observation when the game ends 
        self.reward = 0         # The cummulative reward
        self.location = (1,1)   # The coordinates of the starting chamber
        self.key_location = (0, 0)
        self.key_taken = False
        
    
    def step(self, action: str):
        """
        Takes an action and returns the description for the next state, the reward,
        whether the game is over, and a dictionary of the number of moves and the current score                
        """

        # The game is over
        if self.is_game_over:
            return (self.last_obs, self.reward, self.is_game_over, {'moves' : len(self.moves), 'score' : self.reward})
     
        # Pre-process the action to be recognizable
        if action == "north":
            action = "n"
        elif action == "east":
            action = "e"
        elif action == "south":
            action = "s"
        elif action == "west":
            action = "w"
        elif action == "open the door" or action == "open door":
            action = "open"        

        # Game is not over...process the player's action
        response = self.__process_action(action) 

        # An invalid action (nothing about the state changes)
        if action not in Chambers4Environment.actions:            
            return (response, self.reward, False, {'moves': len(self.moves), 'score' : self.reward})

        # A valid action: record the action
        self.moves.append(action)

        # Did they just win?!
        if response == Chambers4Environment.winning_obs:
            self.is_game_over = True
            self.is_victory = True
            self.last_obs = Chambers4Environment.winning_obs
            self.reward += Chambers4Environment.reward_door            
            return (self.last_obs, self.reward, self.is_game_over, {'moves': len(self.moves), 'score' : self.reward}) 

        # Did they just lose (i.e. this is their last move and they didn't win)
        if len(self.moves) == Chambers4Environment.max_num_moves:                        
            self.is_game_over = True
            self.is_victory = False
            self.last_obs = Chambers4Environment.losing_obs                   
            return (self.last_obs, self.reward, self.is_game_over, {'moves': len(self.moves), 'score' : self.reward})
                
        # Otherwise, it's just a plain ole' turn of the game
        return (response, self.reward, self.is_game_over, {'moves': len(self.moves), 'score' : self.reward})

        

    def get_valid_actions(self):
        """Attempts to generate a set of unique valid actions from the current game state"""
        return deepcopy(Chambers4Environment.actions)

    def game_over(self):
        """Returns true if the game is over and the player has lost"""
        return self.is_game_over and not self.is_victory

    def get_score(self):
        """Returns the integer current game score"""
        return self.reward
        
    def victory(self):
        """Returns true if game is over and the player has won"""
        return self.is_game_over and self.is_victory

    def get_moves(self):
        """Returns the integer number of moves taken by the player in the current episode"""
        return len(self.moves)

    def get_max_score(self):
        """Returns the integer maximum possible score for the game"""
        return Chambers4Environment.reward_key + Chambers4Environment.reward_door

    def reset(self):
        """Reset the environment"""
        self.moves = []        
        self.is_game_over = False
        self.is_victory = False
        self.last_obs = None    # Final observation when the game ends 
        self.reward = 0         # The cummulative reward
        self.location = (1,1)   # The coordinates of the starting chamber
        self.key_location = (0, 0)
        self.key_taken = False        
        return (Chambers4Environment.initial_obs, {'moves': 0, 'score': 0})

    def get_player_location(self):
        """Returns the player's location"""
        return self.location

    def get_state(self):
        """Returns the internal game state"""
        state = {'moves' : deepcopy(self.moves), 'game_over': self.is_game_over, 'victory' : self.is_victory, 'last_obs' : self.last_obs, 'reward': self.reward}        
        return state
    

    def set_state(self, state):
        """Sets the internal game state to the specified state"""
        if 'moves' not in state or 'game_over' not in state or 'victory' not in state or 'last_obs' not in state or 'reward' not in state:
            print('Invalid state passed to method')
            raise ValueError

        if not isinstance(state['moves'], list):
            print('Moves must be a list')
            raise ValueError

        # Reset the game
        self.reset()

        # Play out the actions...this is necessary because the key's location is not part of
        # the state. So we need to simulate the moves taken by the user to ensure the key is
        # in the correct location/state
        for m in state['moves']:
            self.step(m)

        if self.is_game_over != state['game_over']:
            print('Invalid state: game_over value not consistent with list of moves')
            raise ValueError

        if self.is_victory != state['victory']:
            print('Invalid state: victory value not consistent with list of moves')
            raise ValueError

        if self.last_obs != state['last_obs']:
            print('Invalid state: last_obs not consistent with list of moves')
            raise ValueError

        if self.reward != state['reward']:
            print('Invalid state: reward not consistent with list of moves')
            raise ValueError

        
    def get_world_state_hash(self):
        """Get a hash of the current state of the game"""
        return 1
    

    def __process_action(self, action: str):

        # Taking the key
        if action == "take key":
            if self.key_taken:
                return "You have already taken the key"
            if self.key_location == self.location:
                self.key_taken = True
                self.reward += Chambers4Environment.reward_key
                return "You take the key"
            else:
                return "There is no key in this room to take"

        # Dropping the key
        if action == "drop key":
            if not self.key_taken:
                return "You do not have a key to drop"
            else:
                self.key_token = False
                self.key_location = self.location
                return "You drop the key"

        # Looking around
        if action == "look":
            return_str = "You are in a chamber."
            if not self.key_taken and self.key_location == self.location:
                return_str +=  "There is a key on the floor. "
            if self.location == (1, 1):
                return_str += "There is a locked door."
            return return_str

        # Fighting (this is purely for humor)
        if action == "fight":
            return "Violence isn't the answer to this one"
        
        # Opening the door
        if action == "open":
            if self.location != (1, 1):
                return "There is no door in this chamber to open."
            elif not self.key_taken:
                return "The door is locked. You need a key to open it"
            else:
                return Chambers4Environment.winning_obs

        # Movement
        move_choice = None
        if self.location == (0, 0):
            if action == "n" or action == "w":
                move_choice = "invalid"
            elif action == "e":
                self.location = (0, 1)
                move_choice = "east"
            elif action == "s":
                self.location = (1, 0)
                move_choice = "south"

        elif self.location == (0, 1):
            if action == "n" or action == "e":
                move_choice = "invalid"
            elif action == "s":
                self.location = (1,1)
                move_choice = "south"
            elif action == "w":
                self.location = (0, 0)
                move_choice = "west"
        
        elif self.location == (1,0):
            if action == "s" or action == "w":
                move_choice = "invalid"
            elif action == "e":
                self.location = (1,1)
                move_choice = "east"
            elif action == "n":
                self.location = (0, 0)
                move_choice = "north" 

        elif self.location == (1,1):
            if action == "e" or action == "s":
                move_choice = "invalid"
            elif action == "n":
                self.location = (0,1)
                move_choice = "north"
            elif action == "w":
                self.location = (1, 0)
                move_choice = "west" 

        if move_choice == "invalid":
           return "You can't go that way"
        elif move_choice == "north":
            return "You go north and enter another chamber"
        elif move_choice == "east":
            return "You go east and enter another chamber"
        elif move_choice == "south":
            return "You go south and enter another chamber"
        elif move_choice == "west":
            return "You go west and enter another chamber"
    
        # Unrecognized command
        return "That's not a command I recognize."

    def copy(self):
        """Returns a copy of the environment"""
        raise NotImplementedError