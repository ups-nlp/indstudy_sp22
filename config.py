""" Defines global parameters """

from rand import * 

# verbosity of 0 --> nothing prints
# verbosity of 1 --> everything prints
VERBOSITY = 1

# Globally, when a random number generated is needed
# we will use the Python random module
# However, for unit testing, this variable is overwritten
# to instead make deterministic choices
random = PythonRandom()