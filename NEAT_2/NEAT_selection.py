import random
import math

import numpy as np
from NEAT import Individual, initialize_network
from NEAT_speciation import Species

###create test population in species distributed
# species = [Species(Individual(initialize_network()), i, 0) for i in range(5)]
# for specie in species:
#     for _ in range(1):
#         specie.add_member(Individual(initialize_network()))
# species[0].add_member(Individual(initialize_network()))
# species[1].add_member(Individual(initialize_network()))
# species[1].add_member(Individual(initialize_network()))
# species[3].add_member(Individual(initialize_network()))
# species[4].clear_members()
#
# for s in range(len(species)):
#    for l in range(len(species[s].get_members())):
#        species[s].get_members()[l].set_fitness(random.randint(0,10))
# species[3].get_members()[0].set_fitness(random.randint(100,120))


def get_num_individuals(species):
    ind = 0
    for specie in species:
        ind += len(specie.get_members())
    return ind

def highest_pop_score(species):
    highest_score = - 15
    for specie in species:
        if specie.get_highest_fitness() > highest_score:
            highest_score = specie.get_highest_fitness()
        else:
            continue
    return highest_score

def get_all_individuals(species):
    all_members = []
    for specie in species:
        for mem in specie.get_members():
            all_members.append(mem)
    return all_members

def calc_offsprings(species, pop_size):
    """ Returns array with number of offsprings necessary at species id index"""
    #print([len(species[i]) for i in range(len(species))])
    # print('num individuals: ', len(get_all_individuals(species)))
    # print('Number of specieses: ', len(species))
    pop_mean_fitness = 0
    species_fitness_sum = np.zeros(len(species))
    species_offsprings = np.zeros(len(species))
    inds_fertile = 0
    for s in range(len(species)):
        temp_fitness = 0
        l = len(species[s].get_members()) #write def get_members
        #print(l)
        if l > 1:
            for i in range(l):
                temp_fitness += species[s].get_members()[i].get_fitness()/l
                #print('fitt val: ', species[s].get_members()[i].get_fitness())
                #print('adjusted fitt val: ', species[s].get_members()[i].get_fitness()/l)
                inds_fertile+=1
            pop_mean_fitness += temp_fitness
            species_fitness_sum[s] = temp_fitness
            #print('temp fitness species ', species[s].get_id(), ' is: ', temp_fitness)
    if inds_fertile > 0:
        pop_mean_fitness = pop_mean_fitness/inds_fertile
    else:
        pop_mean_fitness = 0
    for s in range(len(species)):
        if len(species[s].get_members()) > 1:
            if pop_mean_fitness != 0:
                species_offsprings[s] = species_fitness_sum[s]/pop_mean_fitness

    #Rounding section
    sp_down = np.floor(species_offsprings)
    # print('Population size: ', pop_size)
    if np.sum(sp_down) == pop_size:
        return sp_down
    else:
        dec = species_offsprings%1
        while np.sum(sp_down) < pop_size:
            m = np.argmax(dec)
            sp_down[m]+=1
            dec[m] = 0
    # print('new offsprings to generate: ', sum(sp_down))
    return sp_down

def fitness_of_list(pop):
    return sum([i.get_fitness() for i in pop])

def choose_parents(pa, off):
    #print(pa)
    parents = []
    for i in range(off):
        if len(pa)==2:
            parents.append(pa)
        elif len(pa) > 2:
            choice = random.sample(pa, 2)
            #choice = random.choices(pa, weights = [j.get_fitness()/fitness_of_list(pa)*100 for j in pa],k=2)
            parents.append(choice)
    return parents

def choose_parents_cross_species(species, offsprings):
    # print('Replacing Individuals: ', offsprings)
    pop = get_all_individuals(species)
    parents = []
    for _ in range(int(offsprings)):
        choice = random.sample(pop,2)
        parents.append(choice)
    return parents

# inds = get_num_individuals(species)
# print('Inidividuals: ', inds)
# offsprings = calc_offsprings(species, inds)
# print(offsprings)
def parent_selection(species):
    parents = []
    inds = get_num_individuals(species)
    # print('Individuals parent selection function: ', inds)
        #check if specie will be extinct
    #print('species after pop, ', [len(species[i]) for i in range(len(species))])
    if len(species)>0:
        offsprings = calc_offsprings(species, inds)
        # print('array of offsprings per species: ', offsprings)
        for s in range(len(species)):
            if offsprings[s] > 0:
                if (species[s].get_evolve() >= 3) and (highest_pop_score(species) > species[s].get_highest_fitness()):
                    #Genocide lets go, generate random parent pairs for as many offsprings as this species would have produced
                    species[s].set_evolve(0)
                    species[s].set_highest_fitness(0)
                    p = choose_parents_cross_species(species, offsprings[s])
                else:
                    p = choose_parents(species[s].get_members(), int(offsprings[s]))
                for j in p:
                    parents.append(j)
    return parents

#print(len(parent_selection(species)))