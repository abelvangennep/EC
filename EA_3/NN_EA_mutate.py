import random
import numpy as np
from NN_EA import *


def mutate(individual):
    individual.set_sigma()
    individual.set_network(individual.get_network() + np.random.normal(0, individual.sigma, len(individual.get_network())))

    return individual
  
