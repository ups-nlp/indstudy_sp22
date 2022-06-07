
import unittest

import config
import rand
config.random = rand.Deterministic()

from environment import ChamberEnvironment
from mcts_node import MCTS_node
from mcts_agent import *

class NodeTest(unittest.TestCase):

    def test_single_node(self):
        """A single node with no parent and no children"""
        n = MCTS_node(None, None, [])

        self.assertTrue(n.is_terminal())

        # TODO: Question, should a leaf node return true/false
        # for is_expanded()? It's not clear
        self.assertTrue(n.is_expanded())
        
        self.assertIsNone(n.get_prev_action())
        self.assertIsNone(n.get_parent())
        self.assertEqual(n.get_sim_value(), 0)
        self.assertEqual(n.get_visited(), 0)
        self.assertEqual(n.get_max_children(), 0)
        self.assertEqual(n.get_children(), [])
        self.assertEqual(n.get_new_actions(), [])

        numTrials = 100
        rand_visited = [random.random() for i in range(numTrials)]
        rand_sim_vals = [random.random() for i in range(numTrials)]

        for i in range(numTrials):
            n.update_visited(rand_visited[i])
            n.update_sim_value(rand_sim_vals[i])

        self.assertEqual(n.get_visited(), sum(rand_visited))
        self.assertEqual(n.get_sim_value(), sum(rand_sim_vals))

    def test_chamber_nodes(self):

        env = ChamberEnvironment(None)
        root = MCTS_node(None, None, env.get_valid_actions(use_parallel=False))

        self.assertFalse(root.is_terminal())
        self.assertFalse(root.is_expanded())
        self.assertEqual(root.get_max_children(), len(env.get_valid_actions(use_parallel=False)))
        self.assertEqual(root.get_new_actions(), env.get_valid_actions(use_parallel=False))


    def test_chamber_expand_node(self):
        env = ChamberEnvironment(None)
        root = MCTS_node(None, None, env.get_valid_actions(use_parallel=False))

        print('Initial parent:')
        print(root)

        print()
        print('New Child:')
        new_node = expand_node(root, env)
        print(new_node)

        print()
        print('Updated parent:')
        print(root)


        print()
        print('New Child:')
        new_node = expand_node(root, env)
        print(new_node)

        print()
        print('Updated parent:')
        print(root)



        print()
        print('New Child:')
        new_node = expand_node(root, env)
        print(new_node)

        print()
        print('Updated parent:')
        print(root)
        



        
        

    



if __name__ == '__main__':
    unittest.main()

