import random
import numpy as np
from NEAT import *


def mutate(individual, mutation_prob):
    for node in individual.get_network():
        weight = node.get_weight()
        if weight > mutation_prob:
            # replace with other random weights
            node.set_weight(np.random.uniform(-1,1))

        if weight < mutation_prob:
            # add/subtract random values
            node.set_weight(weight + np.random.uniform(-1,1))

        if weight == mutation_prob:
            # keep the weight
            node.set_weight(weight)
    return individual

