import random
import math

import numpy as np


def select_population(old_pop, old_fitness, tournament_size, population_size):
    new_pop = np.zeros((population_size, 266))
    fitnesses = []
    for i in range(population_size):
        rand_comp = random.sample(range(tournament_size), int(tournament_size))
        winner = old_pop[rand_comp[0]]
        winner_fitness = old_fitness[rand_comp[0]]
        winner_id = rand_comp[0]
        for j in rand_comp[1:]:
            if old_fitness[j] >  winner_fitness:
                winner = old_pop[j]
                winner_fitness = old_fitness[j]
                winner_id = rand_comp[j]
        fitnesses.append(winner_fitness)
        new_pop[i] = winner
        old_pop = np.delete(new_pop, winner_id, 0)
        old_fitness = np.delete(old_fitness, winner_id, 0)
    
    return new_pop, fitnesses


