import random
from NN_EA import Individual
import numpy as np
import math


def crossover(pop):
    offsprings = np.zeros((60, 266))
    for i in range(60):
        num = random.sample(range(60), 2)

        a = np.random.uniform(0, 1, len(pop[num[0]]))
        offspring = pop[num[0]] * a  + (1 - a) * pop[num[1]]

        offspring[265] = offspring[265] * math.exp(np.random.normal(0, 1/math.sqrt(265)))
        offsprings[i][0:265] = offspring[0:265] + np.random.normal(0, offspring[265], 265)

    new_pop = np.vstack((pop, offsprings))
    return new_pop
