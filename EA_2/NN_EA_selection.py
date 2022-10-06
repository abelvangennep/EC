import random
import math

import numpy as np


def tournament_selection(rand_comp): #tournament selection of size 2
    best_fitness = -100
    for ind in rand_comp:
        if ind.get_fitness() > best_fitness:
            best_fitness = ind.get_fitness()
            #print('Best fitness: ', best_fitness)
            winner = ind
    #print('Winner is: ', winner, ' and it fitness: ', winner.get_fitness())
    return winner

def select_population(pop, offsprings, size):
    end_pop = []
    choose_pop = []
    end_fitnesses = []
    #print(pop)
    for ind in pop:
        choose_pop.append(ind)
    for offspring in offsprings:
        # print(choose_pop)
        choose_pop.append(offspring)
    for i in range(len(pop)):
        rand_comp = random.sample(choose_pop, int(size))
        winner = tournament_selection(rand_comp)
        end_pop.append(winner)
        choose_pop.remove(winner)
    for ind in end_pop:
        end_fitnesses.append(ind.get_fitness())
    return end_pop, end_fitnesses


