from Neat import *

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
    c3 = 1
    # compatibility_threshold = 1

    distance = (c1 * len(excess_genes))/N + (c2 * len(disjoint_genes))/N + c3*W
    return distance

def speciation(population):
    species = [[population[0]]]
    compatibility_threshold = 1.5
    for individual in population[1:]:
        for specie in species:
            print(individual, specie[0])
            check_distance = distance(individual, specie[0])
            if check_distance <= compatibility_threshold:
                specie.append(individual)
            else:
                species.append(individual)
    return species


print(speciation(population))


