import unittest
from mcts_node import Node

class NodeTest(unittest.TestCase):

    def single_node(self):
        """A single node with no parent and no children"""
        n = Node(None, None, None)


