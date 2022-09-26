from Neat import *
import time
import numpy as np

class Species():
    def __init__(self, ind, id, hf):
        self.location = ind
        self.evolve = 0
        self.id = id
        self.highest_fitness = hf
        self.members = [ind]

    def get_location(self):
        return self.location

    def get_id(self):
        return self.id

    def inc_evolve(self):
        self.evolve += 1
    def set_evolve(self,val):
        self.evolve = val
    def get_highest_fitness(self):
        return self.highest_fitness
    def set_highest_fitness(self, val):
        self.highest_fitness = val

    def clear_members(self):
        self.members = []

    def add_member(self,mem):
        self.members.append(mem)

def distance(parent1, parent2):
    parent1_net = parent1.get_network()
    parent2_net = parent2.get_network()
    parent1_innov = []
    parent2_innov = []
    for i in range(len(parent1_net)):
        parent1_innov.append(parent1_net[i].get_innov_id())
    for i in range(len(parent2_net)):
        parent2_innov.append(parent2_net[i].get_innov_id())

    matches = list(set(parent1_innov) and set(parent2_innov))
    W = 0
    for i in matches:
        weight1 = 0
        weight2 = 0
        for j in range(len(parent1_net)):
            if parent1_net[j].get_innov_id() == i:
                weight1 = parent1_net[j].get_weight()
        for k in range(len(parent2_net)):
            if parent2_net[k].get_innov_id() == i:
                weight2 = parent2_net[k].get_weight()
        diff = weight1 - weight2
        W += diff

    parent1_highest_innov = max(parent1_innov)

    disjoint_genes = []
    excess_genes = []
    for i in parent2_innov:
        if i not in parent1_innov and i < parent1_highest_innov:
            disjoint_genes.append(i)
        elif i > parent1_highest_innov:
            excess_genes.append(i)

    N = 0
    if len(parent1_net) > len(parent2_net):
        N += len(parent1_net)
    else:
        N += len(parent2_net)

    c1 = 1
    c2 = 1
    c3 = 0.4
    # compatibility_threshold = 1

    distance = abs((c1 * len(excess_genes))/N + (c2 * len(disjoint_genes))/N + c3*W)
    return distance

def speciation(population, species, highest_species_id, compatibility_threshold=12):
    for specie in species:
        specie.inc_evolve()
        specie.clear_members()
    for individual in population:
        tracker = 0
        for specie in species:
            # print(individual, specie[0])
            check_distance = distance(individual, specie.get_location())
            if check_distance <= compatibility_threshold:
                individual.set_species(specie.get_id())
                if specie.get_highest_fitness() < individual.get_fitness():
                    specie.set_highest_fitness(individual.get_fitness())
                    specie.set_evolve(0)
                    specie.add_member(individual)
                tracker += 1
                break
        if tracker == 0:
            species.append(Species(individual, highest_species_id + 1, individual.get_fitness()))
            individual.set_species(highest_species_id + 1)
            highest_species_id += 1

    #group individuals of same species together
    grouped_inds = [[population[0]]]
    for individual in population[1:]:
        placed = False
        for g in range(len(grouped_inds)):
            if individual.get_species() == grouped_inds[g][0].get_species():
                grouped_inds[g].append(individual)
                placed = True
                break
        if placed == False:
            grouped_inds.append([individual])
    #print([[grouped_inds[i][j].get_species() for j in range(len(grouped_inds[i]))] for i in range(len(grouped_inds)) ])
    return grouped_inds, species, highest_species_id


def calc_avg_dist(pop):
    dist_matrix = np.zeros((len(pop),len(pop)))
    distances = []
    for i in range(len(pop)):
        for j in range(i+1,len(pop)):
            dist_matrix[i,j] = distance(pop[i],pop[j])
            distances.append(dist_matrix[i,j])
    return sum(distances)/len(distances)



