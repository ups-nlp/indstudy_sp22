"""
This file contains the node and state classes for building the game tree with a transposition table.
"""

class Node:

    def __init__(self, parent, prev_act, new_actions):
        self.parent = parent
        self.prev_act = prev_act
        self.children = []
        self.sim_value = 0
        self.visited = 0
        self.max_children = len(new_actions)
        self.new_actions = new_actions


class State:

    def __init__(self) -> None:
        pass