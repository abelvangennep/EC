import random
import math

import numpy as np
from Neat import Individual, initialize_network

###create test population in species distributed
#species = []
#species.append([Individual(initialize_network())])
#species.append([Individual(initialize_network()) for i in range(2)])
#species.append([Individual(initialize_network()) for i in range(3)])
#species.append([Individual(initialize_network()) for i in range(4)])
#print(species)
#f_vals = []
#for s in range(len(species)):
#    for l in range(len(species[s])):
#        species[s][l].set_fitness(random.randint(0,1000))
#        f_vals.append(species[s][l].get_fitness())
#print(f_vals)


def get_num_individuals(species):
    ind = 0
    for s in range(len(species)):
        for l in range(len(species[s])):
            ind+=1
    return ind

def calc_offsprings(species, pop_size):
    #print([len(species[i]) for i in range(len(species))])
    pop_mean_fitness = 0
    species_fitness_sum = []
    species_offsprings = []
    inds = get_num_individuals(species)
    for s in range(len(species)):
        temp_fitness = 0
        l = len(species[s])
        #print(l)
        for i in range(l):
            temp_fitness += species[s][i].get_fitness()/l
        pop_mean_fitness += temp_fitness
        species_fitness_sum.append(temp_fitness)
    pop_mean_fitness = pop_mean_fitness/inds
    for s in range(len(species)):
        if pop_mean_fitness != 0:
            species_offsprings.append(species_fitness_sum[s]/pop_mean_fitness)
        else:
            species_offsprings.append(0)

    #Rounding section
    sp_down = [math.floor(i) for i in species_offsprings]
    if sum(sp_down) == pop_size:
        return sp_down
    else:
        dec = np.array([i % 1 for i in species_offsprings])
        while sum(sp_down) < pop_size:
            m = np.argmax(dec)
            sp_down[m]+=1
            dec[m] = 0
    return sp_down

def fitness_of_list(pop):
    return sum([i.get_fitness() for i in pop])

def choose_parents(pa, off):
    parents = []
    for i in range(off):
        if len(pa)==2:
            parents.append(pa)
        else:
            choice = random.sample(pa, 2)
            #choice = random.choices(pa, weights = [j.get_fitness()/fitness_of_list(pa)*100 for j in pa],k=2)
            parents.append(choice)
    return parents

def parent_selection(species):
    parents = []
    i = 0
    inds = get_num_individuals(species)
    while i < len(species):
        if len(species[i]) <= 1:
            species.pop(i)
            i -= 1
        i += 1
    #print('species after pop, ', [len(species[i]) for i in range(len(species))])
    if len(species)>0:
        offsprings = calc_offsprings(species, inds)
        for s in range(len(species)):
            p = choose_parents(species[s], offsprings[s])
            for j in p:
                parents.append(j)
    return parents

#parents = parent_selection(species)
#print(len(parents))