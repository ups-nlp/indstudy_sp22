"""
Node class for building the game tree
"""\

class Node:
    """Interface for an Node class"""
    def __init__(self, parent, prev_act, new_actions):
        raise NotImplementedError

    def is_terminal(self):
        """ Returns true if the node is terminal """
        raise NotImplementedError

    def add_child(self, child):
        """Add a child to the list of children """
        raise NotImplementedError

    def is_expanded(self):
        """ Returns true if the number of child is equal to the max number of children """
        raise NotImplementedError

    def get_prev_action(self):
        """ Returns the previous action """
        raise NotImplementedError

    def get_parent(self):
        """ Returns the parent Node """
        raise NotImplementedError

    def get_sim_value(self):
        """ Returns the simulated value of the Node """
        raise NotImplementedError

    def get_visited(self):
        """ Returns the quanitiy of visits this node has had """
        raise NotImplementedError

    def get_max_children(self):
        """ Returns the maximum number of children this node could have """
        raise NotImplementedError

    def get_children(self):
        """ Returns a list of all the children """
        raise NotImplementedError

    def get_new_actions(self):
        """ Returns the list of unexplored actions """
        raise NotImplementedError

    def update_sim_value(self, delta):
        """ Updates the simulated value of this node by a specified amount """
        raise NotImplementedError

    def update_visited(self, delta):
        """ Updates the visit count of this node by a specified amount """
        raise NotImplementedError

class MCTS_node(Node):
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
        self.SIM_SCALE = .15
        self.parent = parent
        self.prev_act = prev_act
        self.children = []
        self.sim_value = 0
        self.visited = 0
        self.subtree_size = 1
        self.sim_length_scale = 1
        self.max_children = len(new_actions)
        self.new_actions = new_actions

    def is_terminal(self):
        """ Returns true if the node is terminal
        Returns:
            boolean: true if the max number of children is 0
        """
        return self.max_children == 0
    
    def get_visited(self):
        return self.visited

    def get_sim_value(self):
        return self.sim_value

    def changeLength(self, scalar):
        if scalar < 0:
            self.sim_length_scale = self.sim_length_scale*(1-self.SIM_SCALE)
        
        if scalar >0:
            self.sim_length_scale = self.sim_length_scale*(1+self.SIM_SCALE)

    def update_subtree_size(self):
        self.subtree_size = self.subtree_size+1

    def add_child(self, child):
        """Add a child to the list of children"""
        self.children.append(child)

    def is_expanded(self):
        """ Returns true if the number of child is equal to the max number of children.
        Returns:
            boolean: true if the number of child is equal to the max number of children
        """
        return len(self.children) == self.max_children

    
    def get_prev_action(self):
        """ Returns the previous action """
        return self.prev_act

    def get_parent(self):
        """ Returns the parent Node """
        return self.parent

    def get_sim_value(self):
        """ Returns the simulated value of the Node """
        return self.sim_value

    def get_visited(self):
        """ Returns the quanitiy of visits this node has had """
        return self.visited

    def get_max_children(self):
        """ Returns the maximum number of children this node could have """
        return self.max_children

    def get_children(self):
        """ Returns a list of all the children """
        return self.children
    
    def get_child(self, action):
        """Return the child that results from taking the action"""
        for chil in self.children:
            if chil.get_prev_action() == action:
                return chil
        return self

    def get_new_actions(self):
        """ Returns the list of unexplored actions """
        return self.new_actions

    def update_sim_value(self, delta):
        """ Updates the simulated value of this node by a specified amount """
        self.sim_value += delta

    def update_visited(self, delta):
        """ Updates the visit count of this node by a specified amount """
        self.visited += delta