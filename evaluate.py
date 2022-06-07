""" Subroutines for evaluating agents """

import argparse
import config
from agent import RandomAgent
from agent import HumanAgent
from agent import MonteAgent
# from dep_agent import DEPagent
from environment import *
from play import play_game



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluates an agent')

    parser.add_argument('num_trials', type=int,
                        help='Number of times to run the agent on the specified game')
    parser.add_argument('num_moves', type=int,
                        help='Number of moves for agent to take per trial')
    parser.add_argument('agent', help='[random|human|mcts|dep]')
    parser.add_argument('game', help='[path to game file|chamber|chamber4]')
    parser.add_argument('num_seconds',help='number of seconds agent gets to make a move')
    parser.add_argument('num_trees',help='number of trees to build with parallel mcts')
    parser.add_argument('-v', '--verbosity', type=int,
                        help='[0|1] verbosity level')
    args = parser.parse_args()
    ready = 0
    args.num_trees = int(args.num_trees)
    args.num_seconds = int(args.num_seconds)
    if args.num_trees != 4 and args.num_trees != 8:
        print("select [4|8] for number of trees")
        ready = 1
    if args.num_seconds != 20 and args.num_seconds != 40 and args.num_seconds != 60:
        print(args.num_seconds)
        print("select [20|40|60] for number of seconds")
        ready = 1

    # Instantiate the game environment    
 

    # Set the verbosity level
    if args.verbosity == 0 or args.verbosity == 1:
        config.VERBOSITY = args.verbosity

    total_score = 0                 # total agent score aggregated over all trials
    total_num_valid_actions = 0     # total number of valid actions aggregated over all trials
    total_num_location_changes = 0  # total number of location changes aggregated over all trials
    total_num_steps = 0             # total number of steps taken aggregated over all trials
                                    # this may be less than num_steps * num_trials if the agent dies or wins
                                    # before the num_steps is up
    total_time = 0                  # total seconds taken aggregated over all trials


    if ready == 0:
        for i in range(args.num_trials):
            if args.game == "chamber":
                    env = ChamberEnvironment(None)
            elif args.game == "chamber4":
                    env = Chambers4Environment(None)
            else:
                    # args.game is the path name to a Z-master game
                    env = JerichoEnvironment(args.game)

            # Instantiate the agent
            if args.agent == 'random':
                ai_agent = RandomAgent()
            elif args.agent == 'human':
                ai_agent = HumanAgent()
            elif args.agent == 'mcts':
                ai_agent = MonteAgent(env, args.num_moves, args.num_seconds, args.num_trees)
            # elif args.agent == 'dep':
                # ai_agent = DEPagent()
            else:
                ai_agent = RandomAgent()
            score, num_valid_actions, num_location_changes, num_steps, time = play_game(
                ai_agent, env, args.num_moves)
            env.close()

            total_score += score
            total_num_valid_actions += num_valid_actions
            total_num_location_changes += num_location_changes
            total_num_steps += num_steps
            total_time += time

            file_str = f'parallelTesting/{args.num_trees}trees/{args.num_trees}t{args.num_seconds}s.txt'
            data_file = open(file_str,"a")
            new_line = f'{score}\t{num_steps}\t{num_valid_actions}\t{num_location_changes}\t{time}\n'
            data_file.write(new_line)
            data_file.close()
            print(f'Trial {i+1}:')
            print(f'Score= {score}')
            print(f'Number of steps: {num_steps} out of a possible {args.num_moves}')
            print(f'Of those {num_steps} steps, how many were valid? {num_valid_actions}')
            print(f'Of those {num_steps} steps, how many changed your location? {num_location_changes}')
            print(f'How long to call take_action() {num_steps} times? {time}')
            print()

        print()
        print(f'Number of trials: {args.num_trials}')
        print(f'Number of moves per trial: {args.num_moves}')
        print()
        print(f'Total number of steps: {total_num_steps} across {args.num_trials} trials')
        print(f'Max number of steps possible: {args.num_trials * args.num_moves}')
        print(f'Average number of steps per trial: {total_num_steps/args.num_trials}')
        print(f'Average score: {total_score/args.num_trials}')
        print(f'Average num valid steps: {total_num_valid_actions/args.num_trials}')
        print(f'Average location changes: {total_num_location_changes/args.num_trials}')
        print(f'Total time taken calling take_action(): {total_time}')
        print(f'Avg. seconds for take_action(): {total_time/total_num_steps}')
