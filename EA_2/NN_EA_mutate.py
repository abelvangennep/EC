import random
import numpy as np
from NN_EA import *


def mutate(individual, mutation_prob):
    for node in individual.get_network():
        r = np.random.uniform(0, 1)
        if r > 1 - mutation_prob:
            a = np.random.uniform(0, 1)
            if a < 1 / 3:
                # replace with other random weights
                node.set_weight(np.random.uniform(-1, 1))
            elif a > 2 / 3:
                node.set_weight(node.get_weight() + np.random.uniform(-1, 1))
            else:
                # switch weights
                switch = random.randint(0, len(individual.get_network()) - 1)
                temp = individual.get_network()[switch].get_weight()
                individual.get_network()[switch].set_weight(node.get_weight())
                node.set_weight(temp)
    return individual


def mutate(individual):
    for connection in individual.get_network():
        connection.set_weight(connection.get_weight() + np.random.normal(0, individual.sigma))

    individual.set_sigma()

    return individual
  
