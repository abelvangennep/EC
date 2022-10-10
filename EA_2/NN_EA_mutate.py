import random
import numpy as np
from NN_EA import *


def mutate(individual):
    for connection in individual.get_network():
        connection.set_weight(connection.get_weight() + np.random.normal(0, individual.sigma))

    individual.set_sigma()

    return individual
  
