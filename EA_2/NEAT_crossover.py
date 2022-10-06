import random
from NEAT import Individual, Connection_Gene

def crossover(pop):
    parents = random.sample(pop,2)
    offsprings = []
    for _ in range(len(pop)):
        child = []
        for i in range(len(parents[0].get_network())):
            a = random.uniform(0, 1)
            conn = parents[0].get_network()[i]
            conn.set_weight(parents[0].get_network()[i].get_weight()*a +(1-a) * parents[1].get_network()[i].get_weight())
            child.append(conn)
        offsprings.append(Individual(child))
    return offsprings




