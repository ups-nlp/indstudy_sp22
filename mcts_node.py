"""
Node class for building the game tree
"""

class Node:
    """
    This Node class represents a state of the game. Each node holds the following:
    parent -- it's parent node
    prev_act -- the previous action taken to get to this node
    children -- a list of the children of this node
    sim_value -- the simulated value of the node
    visited -- the number of times this node has been visited
    max_children -- the total number of children this node could have
    new_actions -- a list of the unexplored actions at this node

    Keyword arguments:
    parent -- it's parent node
    prev_act -- the previous action taken to get to this node
    new_actions -- a list of all the unexplored actions at this node
    """

    def __init__(self, parent, prev_act, new_actions):
        self.parent = parent
        self.prev_act = prev_act
        self.children = []
        self.sim_value = 0
        self.visited = 0
        self.max_children = len(new_actions)
        self.new_actions = new_actions

    def is_terminal(self):
        """ Returns true if the node is terminal
        Returns:
            boolean: true if the max number of children is 0
        """
        return self.max_children == 0

    def print(self, level):
        """ Print a text representation of the tree
        """
        space = ">" * level
        #for i in range(level):
        #    space += ">"
        if self.prev_act is None:
            print("\t"+space+"<root>"+"\n")
        else:
            print("\t"+space+self.prev_act+"\n")

        for child in self.children:
            child.print(level+1)

    def add_child(self, child):
        """Add a child to the list of children"""
        self.children.append(child)

    def get_parent(self):
        """Return the node's parent in the tree"""
        return self.parent

    def get_prev_action(self):
        """Get the action that led to this node"""
        return self.prev_act

    def get_children(self):
        """Return the list of children"""
        return self.children

    def is_expanded(self):
        """ Returns true if the number of child is equal to the max number of children.
        Returns:
            boolean: true if the number of child is equal to the max number of children
        """
        return len(self.children) == self.max_children