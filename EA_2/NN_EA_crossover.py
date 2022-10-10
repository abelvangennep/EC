import random
from NN_EA import Individual


def crossover(pop):
    parents = random.sample(pop, 2)
    if parents[0].get_fitness() > parents[1].get_fitness():
        strongest_parent = parents[0]
        weak_parent = parents[1]
    else:
        strongest_parent = parents[1]
        weak_parent = parents[0]

    offsprings = []
    
    for _ in range(len(pop)):
        child = []
        for i in range(len(parents[0].get_network())):
            a = random.uniform(0, .5)
            conn = strongest_parent.get_network()[i]
            conn.set_weight(
                weak_parent.get_network()[i].get_weight() * a + (1 - a) * strongest_parent.get_network()[i].get_weight())
            child.append(conn)

        sigma = (weak_parent.sigma *  a + (1 - a) * strongest_parent.sigma) /  2

        offsprings.append(Individual(child, sigma))

    return offsprings
