import random
from NEAT import Individual, Connection_Gene

def crossover(parent_1, parent_2):
    child = []
    for i in range(len(parent_1.get_network())):
        a = random.uniform(0, 1)
        conn = parent_1.get_network()[i]
        conn.set_weight(parent_1.get_network()[i].get_weight()*a +(1-a) * parent_2.get_network()[i].get_weight())
        child.append(conn)
    return Individual(child)




