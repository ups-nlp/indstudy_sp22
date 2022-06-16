"""
Node class for building the game tree
"""

# TODO: Store env.game_over() for each node so we know if a node corresponds to a terminal state or not

class Node:
    """Interface for an Node class"""
    def __init__(self, parent, prev_act, new_actions, score):
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

    def get_score(self):
        """ Returns the score of the game at this Node"""
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

    def remove_action(self, action):
        """ Returns an action from the list of unexplored actions """
        raise NotImplementedError

    def update_sim_value(self, delta):
        """ Updates the simulated value of this node by a specified amount """
        raise NotImplementedError

    def update_visited(self, delta):
        """ Updates the visit count of this node by a specified amount """
        raise NotImplementedError

    def __str__(self):
        """Returns a string representation of the node"""
        raise NotImplementedError


class MCTS_node(Node):
    """
    This Node class represents a state of the game. Each node holds the following:
    parent -- its parent node
    prev_act -- the previous action taken to get to this node
    score -- the score of the game at this node
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

    def __init__(self, parent, prev_act, new_actions, score):
        
        # Although it's okay for parent and prev_act to be None
        # it is never okay for new_actions to be None
        # If this happens, we replace it with an empty list
        if new_actions is None:
            new_actions = [] 

        self.parent = parent
        self.score = score
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

    def get_score(self):
        """ Returns the score of the game at this Node"""
        return self.score

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

    def get_new_actions(self):
        """ Returns the list of unexplored actions """
        return self.new_actions

    def remove_action(self, action):
        """ Returns an action from the list of unexplored actions """
        self.new_actions.remove(action)

    def update_sim_value(self, delta):
        """ Updates the simulated value of this node by a specified amount """
        self.sim_value += delta

    def update_visited(self, delta):
        """ Updates the visit count of this node by a specified amount """
        self.visited += delta

    def __str__(self):
        child_node_str = "["
        for child in self.children:
            child_node_str += child.prev_act + ", "            
        child_node_str += "]"
        
        if self.parent is None:
            return f'[Parent: Null, prev_act: {self.prev_act}, sim_value: {self.sim_value}, visited: {self.visited}, max_children: {self.max_children}, new_actions:{self.new_actions}, children: {child_node_str}]\n'
        else:
            return f'[Parent: {self.parent.prev_act}, prev_act: {self.prev_act}, sim_value: {self.sim_value}, visited: {self.visited}, max_children: {self.max_children}, new_actions:{self.new_actions}, children: {child_node_str}]\n'
        
        