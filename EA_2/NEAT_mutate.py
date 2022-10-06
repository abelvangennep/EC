import random
import numpy as np


def mutate(individual, mutation_prob):
    for n in individual:
        r = np.random.uniform(-1,1)
        if n > mutation_prob:
            # replace with other random weights
            n.set_weight(np.random.uniform(-1,1))

        if n < mutation_prob:
            # add/subtract random values
            n.set_weight(n.get_weight + r)

        if n == mutation_prob:
            # switch with other weights

            weight = get_weight()
            n.set_weight(weight)
    id+=1
    return individual

