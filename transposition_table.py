"""
This file contains the node and state classes for building the game tree with a transposition table.
"""

def get_world_state_hash(location, valid_actions):
    #print(location+str(valid_actions))
    return location+str(valid_actions)

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

    def __init__(self, state, parent, prev_act, new_actions, transposition_table):
        self.parent = parent
        self.prev_act = prev_act
        self.state = state

        if(transposition_table.get(state) is None):
            # Create a key-value pair for the new state
            transposition_table[state] = State(parent, new_actions)
        else:
            # Update the existing key-value pair for the state
            entry = transposition_table.get(state)
            entry.parents.append(parent)
            #entry.children = []
            #entry.new_actions.remove(prev_act)
        
        self.state = transposition_table.get(state)



    def is_terminal(self):
        """ Returns true if the node is terminal
        Returns:
            boolean: true if the max number of children is 0
        """
        return self.state.max_children == 0


    def add_child(self, child):
        self.state.children.append(child)

    def is_expanded(self):
        """ Returns true if the number of child is equal to the max number of children.
        Returns:
            boolean: true if the number of child is equal to the max number of children
        """
        return (len(self.state.children) >= self.state.max_children)

    
    def get_prev_action(self):
        return self.prev_act

    def get_parent(self):
        return self.parent

    # The below methods fetch the fields from the Node's State

    def get_state(self):
        return self.state

    def get_sim_value(self):
        return self.state.sim_value

    def get_visited(self):
        return self.state.visited
    
    def get_state_parents(self):
        return self.state.parents

    def get_max_children(self):
        return self.state.max_children

    def get_children(self):
        return self.state.children

    def get_new_actions(self):
        return self.state.new_actions

    
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
    parents -- The set of parents leading to this state
    max_children -- the total number of children this node could have
    children -- a list of the explored states from this node
    new_actions -- a list of the unexplored actions at this node

    Keyword arguments:
    parent -- a parent state taken to get to this node
    new_actions -- a list of all the unexplored actions at this node
    """

    def __init__(self, parent, new_actions):
        # The simulated value for this state of the game
        self.sim_value = 0
        # The number of times we have simulated this state of the game
        self.visited = 0
        # The set of parents leading to this state
        self.parents = [parent]
        # the total number of children this state can have
        self.max_children = len(new_actions)
        # a list of the explored states from this node
        self.children = []
        # a list of the unexplored actions at this node
        self.new_actions = new_actions

 
