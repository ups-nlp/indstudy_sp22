#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 21.10.30
Recreated 22.2.15

@author: prowe
@adapted for general use by eric markewitz

A test player to get us the walkthroughs from all jericho games
"""
import os
import re

from jericho import FrotzEnv

# Our modules
#import config

currVerbosity = 0

def set_up_game(game_file: str):

    #create the environment
    env = FrotzEnv(game_file)

    # The history is a list of (observation, action) tuples
    history = []

    # Get the initial observation and info
    # info is a dictionary (i.e. hashmap) of {'moves':int, 'score':int}
    curr_obs, info = env.reset()

    return curr_obs, history, env

def game_step(action_to_take, curr_obs, history, env):
    history.append((curr_obs, action_to_take))
    next_obs, _, done, info = env.step(action_to_take)
    curr_obs = next_obs

    if currVerbosity > 0:
        print('\n\n=========================================')
        print('Taking action: ', action_to_take)
        print('Game State:', next_obs.strip())
        print('Total Score', info['score'], 'Moves', info['moves'])

    return curr_obs, history, env



def print_status(env, history):
    print_history(history)
    print_inventory(env)

def print_inventory(env):
    print('\n\n============= INVENTORY =============')
    for item in env.get_inventory():
        print(item.name)

def print_history(history):
    print('\n\n============= HISTORY  =============')
    for obs, action in history:
        print(obs, "   ---> Action:", action)

def print_actions(history):
    print('\n\n============= HISTORY OF ACTIONS TAKEN =============')
    for _, action in history:
        print(action)


def get_walkthrough(game_file, walkthrough_source):
    """
    Play the game for a walkthrough and get the accompanying
    observations
    """


    # Set up and play the game
    curr_obs, history, env = set_up_game(game_file)

    # Get the actions - this could be, e.g. from a file. Here it is from
    # the built-in walkthrough
    if walkthrough_source == 'builtin':
        actions = env.get_walkthrough()
    else:
        raise ValueError('No other walkthroughs available yet')

    walkthrough = []
    # Take the actions, one by one
    while len(actions) > 0:
        prev_score = env.get_score()
        action_to_take = actions.pop(0)
        curr_obs, history, env = game_step(action_to_take, curr_obs, history, env)
        points = env.get_score() - prev_score
        walkthrough.append((history[-1][0], history[-1][1], points))

    if currVerbosity > 0:
        print_actions(history)
        print_history(history)
        print_inventory(env)

    #print('\nConfirm I won:', env.victory())

    return walkthrough




def categorize_action(action:str) -> int:
    """
    Looks at the action and sees which module it came from

    @param a single action
    @return the module it came from
    """

    movements = {'north','south','east','west','up','down','northwest','southeast','northeast', 'southwest', 'go', 'climb', 'jump'}

    action = action.split(' ')
    action_set = set(action)


    #sometimes categorizes actions as movements when they use an action word as a descriptor like 'press west wall'
    if bool(action_set & movements):
        return 0

    else:
        return 1





if __name__ == "__main__":
    """
    Create a walkthrough as a csv file of observation, action
    for a given file with a list of actions, or the builtin walkthrough

    """
    # # # # # # # # # # # #   INPUTS    # # # # # # # # # # # # # # # #
    # The location of your Jericho game suite. This is unique to your computer,
    # so you should modify it
    games_dir = '../../z-machine-games-master/jericho-game-suite/'

    game_files = os.listdir(games_dir)
    game_files.sort()

    lines = []

    corpus = ''


    for game in game_files:
        if game.count(".z5") >0:
            game_file_path = games_dir + game

            # The source of the actions
            walkthrough_source = 'builtin'

            walkthrough = get_walkthrough(game_file_path, walkthrough_source)

            #print(game)
            #print()
            #print("--------------------------")
            for line in walkthrough:


                obs = line[0]
                action = line[1]
                score = line[2]


                obs = obs.casefold()
                action = action.casefold()

                obs = re.sub('\n',' ',obs)
                obs = re.sub('[\[\]?+-=*,.|\:;\(\)!\"\>\']','', obs)
                #obs = re.sub('\[','', obs)
                obs = re.sub(r'\s+',' ', obs)
                obs = obs.strip()

                action = re.sub('[\[\]?+-=*,.|\:;\(\)!\"\>\']','', action)
                action = re.sub(r'\s+',' ', action)
                action = action.strip()

                if action == "n":
                    action = "north"
                elif action == "s":
                    action = "south"
                elif action == "e":
                    action = "east"
                elif action == "w":
                    action = "west"
                elif action == "ne":
                    action = "northeast"
                elif action == "nw":
                    action = "northwest"
                elif action == "se":
                    action = "southeast"
                elif action == "sw":
                    action = "southwest"
                elif action == "u":
                    action = "up"
                elif action == "d":
                    action = "down"


                action_type = categorize_action(action)


                #trainingTuple = (obs, action, action_type)
                trainingLine = obs + "," + action + "," + str(action_type) + "\n"
                lines.append(trainingLine)

                corpusLine = obs + " " + action + " "
                corpus += corpusLine

                #print(trainingTuple)
                #print()



                #print(action)
                #print(action_type)


                #print(score)


                #for thing in line:
                    #strThing = str(thing)
                    #strThing = re.sub('\n',' ',strThing)
                    #print(strThing)
                    #print()

            #print()
            #print()
            #print()

    #print(game_files)

    training_data = open("walkthrough_training_data.txt", "w")
    training_data.writelines(lines)

    walkthrough_corpus = open("walkthrough_corpus.txt", "w")
    walkthrough_corpus.write(corpus)

    """
    # The name of the game you want to play
    game_file = jericho_dir + 'zork1.z5'

    # The source of the actions
    walkthrough_source = 'builtin'

    # The file to save to
    outfile = 'data/frotz_builtin_walkthrough.csv'
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #


    # Get a list of tuples of observation, action
    walkthrough = get_walkthrough(game_file, walkthrough_source)

    # Write the file
    with open(outfile,'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['Observation', 'Action', 'Points'])
        for row in walkthrough:
            csv_out.writerow(row)
    """
