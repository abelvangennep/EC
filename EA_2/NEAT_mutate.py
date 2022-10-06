import random
import numpy as np

def mutate(individual, mutation_prob):
    for n in individual.get_network():
        r = np.random.uniform(0,1)
        if n > mutation_prob:
            n.set_weight(np.random.uniform(-1,1))
            #switch with other weight
            #replace by new weight
            #add random value to current weight
    return individual
