import random
from Neat import Individual, Connection_Gene

def crossover(parent_1, parent_2):
    #check which parent is superior
    if parent_1.get_fitness() > parent_2.get_fitness():
        superior_parent = parent_1
        inferior_parent = parent_2

    elif parent_1.get_fitness() < parent_2.get_fitness():
        superior_parent = parent_2
        inferior_parent = parent_1

    elif random.choice([True, False]):
        superior_parent = parent_1
        inferior_parent = parent_2
    
    else:
        superior_parent = parent_2
        inferior_parent = parent_1

    child = []

    superior_id_tracker  = 0
    inferior_id_tracker = 0
    superior_network = superior_parent.get_network()
    inferior_network = inferior_parent.get_network()

    for _ in range(superior_network + inferior_network):
        if superior_network[superior_id_tracker].innov_id == inferior_network[inferior_id_tracker].innov_id:
            if superior_network[superior_id_tracker].enabled and inferior_network[inferior_id_tracker].enabled:
                child.append(Connection_Gene(superior_network[superior_id_tracker].in_node, superior_network[superior_id_tracker].out_node, superior_network[superior_id_tracker].weight, superior_network[superior_id_tracker].innov_id, True))
            else:
                child.append(Connection_Gene(superior_network[superior_id_tracker].in_node, superior_network[superior_id_tracker].out_node, superior_network[superior_id_tracker].weight, superior_network[superior_id_tracker].innov_id, False))
            superior_id_tracker += 1

        elif superior_network[superior_id_tracker].innov_id > inferior_network[inferior_id_tracker].innov_id:
            child.append(Connection_Gene(superior_network[superior_id_tracker].in_node, superior_network[superior_id_tracker].out_node, superior_network[superior_id_tracker].weight, superior_network[superior_id_tracker].innov_id, superior_network[superior_id_tracker].enabled))
            inferior_id_tracker += 1

        elif superior_network[superior_id_tracker].innov_id < inferior_network[inferior_id_tracker].innov_id:
            child.append(Connection_Gene(inferior_network[superior_id_tracker].in_node, inferior_network[superior_id_tracker].out_node, inferior_network[superior_id_tracker].weight, inferior_network[superior_id_tracker].innov_id, inferior_network[superior_id_tracker].enabled))
            superior_id_tracker += 1

        else:
            print("if we get here something is going wrong in NEAT crossover")
            break

    return Individual(child)




