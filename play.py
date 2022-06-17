"""Instantiates an AI agent to play the specified game"""

import argparse
import sys
import time
from os.path import exists

from agent import Agent
from agent import RandomAgent
from agent import HumanAgent
from agent import MonteAgent
import config
from environment import *



def play_game(agent: Agent, env: Environment, num_steps: int, file_str):
    """ The main method that instantiates an agent and plays the specified game"""


    # =============== DELETE WHEN DONE ============================
    file_str = file_str.split('.')[0] + '_states' + '.txt'
    # Open file
    if exists(file_str):        
        data_file = open(file_str, "a", buffering=1)
    else:
        data_file = open(file_str, "w", buffering=1)
    # ===========================================================


    # The history is a list of (observation, action) tuples
    history = []

    # Reset the environment
    curr_obs, info = env.reset()    
    done = False

    if config.VERBOSITY > 0:
        print('=========================================')
        print("Initial Observation\n" + curr_obs)

    prev_location = env.get_player_location() 
    num_location_changes = 0  # total number of times an action led to a change in location
    num_times_called = 0 # total number of times take_action() was called
    num_iters = 0 # inside of take_action(), total number of MCTS iterations performed
    seconds = 0 # total time spent in take_action() over all iterations
    
    curr_score = env.get_score() # current score of the game
    distance = 0 # distance between scoring actions
    dist_btw_scores = [] # distance between scoring states

    while num_steps != 0 and not done:
        if config.VERBOSITY > 0:
            print('\n\n=========================================')

        # timing the call to take_action()
        start_time = time.time()
        action_to_take, iters, num_states = agent.take_action(env, history)
        end_time = time.time()


        # ================= DELETE WHEN DONE =====================
        new_line = f'{iters}\t{num_states}\n'
        data_file.write(new_line)
        # ========================================================

        # updating statistics
        num_times_called += 1
        num_iters += iters      # THIS IS ALSO THE TOTAL NUMBER OF NODES EVER CREATED
        seconds += (end_time - start_time)

        # updating environment with selected action
        next_obs, _, done, info = env.step(action_to_take)        
        history.append((curr_obs, action_to_take))

        # checking if the action taken caused a change in location
        curr_location = env.get_player_location()
        if prev_location != curr_location:
            num_location_changes += 1
        prev_location = curr_location

        # checking if the action taken caused a change in score
        next_score = env.get_score()
        if next_score == curr_score:
            distance += 1
        else:
            dist_btw_scores.append(distance)
            distance = 0

        # updating variables as needed
        curr_score = next_score
        curr_obs = next_obs

        if config.VERBOSITY > 0:
            print()
            print('Taking action: ', action_to_take)
            print('Game State:', next_obs.strip())
            print('Total Score', info['score'], 'Moves', info['moves'])

        if num_times_called > 0 and num_times_called % 10 == 0:
            print()
            print('\t====== PARTIAL REPORT ======')
            s = info['score']
            m = info['moves']
            print(f'\tScore= {s}')
            print(f'\tNumber of steps so far: {num_times_called}')            
            print(f'\tOf those {num_times_called} steps, how many were valid? {m}')
            print(f'\tOf those {num_times_called} steps, how many changed your location? {num_location_changes}')
            print(f'\tHow long to call take_action() {num_times_called} times? {seconds}')
            print(f'\tAverage number of seconds spent in take_action() {seconds/num_times_called}')
            print(f'\tNumber of MCTS iterations performed overall? {num_iters}')
            print(f'\tAverage number of MCTS iters per call to take_action()? {num_iters/num_times_called}')
            print()

        num_steps -= 1


    data_file.write('\n')
    data_file.close()

    if config.VERBOSITY > 1:
        print('\n\n============= HISTORY OF ACTIONS TAKEN =============')
        for _, action in history:
            print(action)

    # Computing average distance between score changes
    avg_dist = 0
    if len(dist_btw_scores) > 0:
        avg_dist = sum(dist_btw_scores)/len(dist_btw_scores)

    return (info['score'], info['moves'], num_location_changes, num_times_called, seconds, num_iters, avg_dist)


if __name__ == "__main__":
    # Read in command line arguments and play the game with the specified parameters
    # Uses a parser for the command line arguments:
    # num_moves -- The number of moves the agent should make
    # agent -- Right now this is just 'random' but will expand as we make other agents
    # game_file -- The full path to the game file

    parser = argparse.ArgumentParser(
        description='Runs an AI agent on a specified game')

    # Positional arguments are required by default
    parser.add_argument(
        'num_moves', type=int, help="Number of moves for the agent to make. Enter '-1' for unlimited moves.")
    parser.add_argument('agent', help='[random|human|mcts]')    
    parser.add_argument('game', help='[path to game file|chamber|chamber4]')

    # Optional arguments are...optional
    parser.add_argument('-t' , '--mcts_time', type=int, help='Number of seconds to run MCTS algorithm before choosing an action')
    parser.add_argument('-d' , '--mcts_depth', type=int, help='Maximum depth of any path generated by the default policy for the MCTS algorithm')
    parser.add_argument('-v', '--verbosity', type=int,
                        help='[0|1] verbosity level')
    args = parser.parse_args()

    # Instantiate the game environment    
    if args.game == "chamber":
        env = ChamberEnvironment(None)
    elif args.game == "chamber4":
        env = Chambers4Environment(None)
    else:
        # args.game is the path name to a Z-master game
        env = JerichoEnvironment(args.game)
        
    # Create the agent
    if args.agent == 'random':
        ai_agent = RandomAgent()
    elif args.agent == 'human':
        ai_agent = HumanAgent()    
    elif args.agent == 'mcts':
        if args.mcts_time is None:
            print('Error: must set the mcts_time limit')
            sys.exit()
        elif args.mcts_depth is None:
            print('Error: must set mcts_depth')
            sys.exit()
        else:
            # Note: We are creating a JerichoEnvironment here as well as below in the play() method
            # The JerichoEnvironment we are passing in to the MonteAgent is simply used to get a list
            # of starting actions. Once the constructor is finished, this environment object is never
            # used again. Going forward, we should think of a way to all use the same environment
            # from the start onwards. 
            ai_agent = MonteAgent(env, args.mcts_time, args.mcts_depth)
    else:
        ai_agent = RandomAgent()

    # Set the verbosity level
    if args.verbosity is not None and (0 <= args.verbosity and args.verbosity <= 2):        
        config.VERBOSITY = args.verbosity

    play_game(ai_agent, env, args.num_moves)
