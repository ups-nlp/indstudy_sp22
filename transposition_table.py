"""
This file contains the node and state classes for building the game tree with a transposition table.
"""


def get_world_state_hash(location, valid_actions):
    return str(location)+str(valid_actions)


class Transposition_Node:
    """
    This Node class maps to a state of the game. 
    Each node holds the following:
    state -- the state of the game represented by the node
    parent -- it's parent node
    prev_act -- the previous action taken to get to this node

    Keyword arguments:
    state -- the state of the game represented by the node
    parent -- it's parent node
    prev_act -- the previous action taken to get to this node
    new_actions -- a list of all the unexplored actions at this node
    transposition_table -- the HashMap to the transposition table
    """

    def __init__(self, state, parent, prev_act, new_actions, transposition_table, score):

        # Although it's okay for parent and prev_act to be None
        # it is never okay for new_actions to be None
        # If this happens, we replace it with an empty list
        if new_actions is None:
            new_actions = []

        self.parent = parent
        self.prev_act = prev_act
        self.score = score
        self.children = []
        self.max_children = len(new_actions)
        self.new_actions = new_actions

        if(transposition_table.get(state) is None):
            # Create a key-value pair for the new state
            transposition_table[state] = State(state)
        self.state = transposition_table.get(state)
        self.state.increment_usage()

    def toString(self):
        if(self.parent is not None):
            return self.state.toString()  # +" with parent: "+self.parent.state.toString()
        else:
            return self.state.toString()  # +" with NONE parent!"

    def __str__(self):
        child_node_str = "["
        for child in self.children:
            child_node_str += child.prev_act + ", "
        child_node_str += "]"

        if self.parent is None:
            return f'[Parent: Null, prev_act: {self.prev_act}, sim_value: {self.state.sim_value}, visited: {self.state.visited}, max_children: {self.max_children}, new_actions:{self.new_actions}, children: {child_node_str}]\n'
        else:
            return f'[Parent: {self.parent.prev_act}, prev_act: {self.prev_act}, sim_value: {self.state.sim_value}, visited: {self.state.visited}, max_children: {self.max_children}, new_actions:{self.new_actions}, children: {child_node_str}]\n'

    def is_terminal(self):
        """ Returns true if the node is terminal
        Returns:
            boolean: true if the max number of children is 0
        """
        return self.max_children == 0

    def add_child(self, child):
        """Add a child to the list of children"""
        self.children.append(child)

    def is_expanded(self):
        """ Returns true if the number of child is equal to the max number of children.
        Returns:
            boolean: true if the number of child is equal to the max number of children
        """
        # return (len(self.children) >= self.max_children)
        return len(self.children) == self.max_children

    def get_prev_action(self):
        return self.prev_act

    def get_parent(self):
        return self.parent

    def get_score(self):
        return self.score

    def get_max_children(self):
        return self.max_children

    def get_children(self):
        return self.children

    def get_new_actions(self):
        return self.new_actions

    def remove_action(self, action):
        """ Returns an action from the list of unexplored actions """
        self.new_actions.remove(action)

    # The below methods fetch the fields from the Node's State

    def get_state(self):
        return self.state

    def get_sim_value(self):
        return self.state.sim_value

    def get_visited(self):
        return self.state.visited

    # The below methods updates the fields from the Node's State

    def update_sim_value(self, delta):
        self.state.sim_value += delta

    def update_visited(self, delta):
        self.state.visited += delta


class State:
    """
    This State class represents a state of the game. 
    Each entry holds the following:
    sim_value -- the simulated value of the node
    visited -- the number of times this node has been visited
    """

    def __init__(self, state):
        # The simulated value for this state of the game
        self.sim_value = 0

        # The number of times we have simulated this state of the game
        self.visited = 0

        # QUESTION: Why is the state stored in the state? The state is a str that is a
        # key into the hashmap. When you use the state (i.e. the key) to index into the hashmap
        # what is returned is this state object. Is there a reason we need to keep the key
        # with the object (i.e. the value)?
        self.state = state

        # Counts the number of nodes that map to this state
        self.usage = 0

    def toString(self):
        return str(self.state)+" has been explored " + str(self.visited) + " times and has a score of " + str(self.sim_value)

    def increment_usage(self):
        self.usage += 1

    def get_usage(self):
        return self.usage
