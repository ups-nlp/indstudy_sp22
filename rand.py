import random

"""
Defines an interface for a random number generator.

The two implementing classes are:
- A wrapper for the Python module
- A class that returns deterministic choices for the purposes of unit testing
"""

class Rand:
    """Interface for a Random Number Generator"""

    def choice(self, choices: list):
        """Returns a possibly randomly selected choice from the list of choices"""
        raise NotImplementedError

    def random(self):
        """Returns a random floating point number in the range [0.0, 1.0)"""
        raise NotImplementedError



class PythonRandom(Rand):
    """
    This class simply calls the random.choice() method from 
    the random module
    """
    def choice(self, choices: list):
        return random.choice(choices)

    def random(self):
        """Returns a random floating point number in the range [0.0, 1.0)"""
        return random.random()



class Deterministic(Rand):
    """
    This class is not random. Instead it deterministically returns the first
    element in any list that is passed in.     
    """

    def choice(self, choices: list):
        return choices[0]

    def random(self):
        """Returns a random floating point number in the range [0.0, 1.0)"""
        return 0.0


    