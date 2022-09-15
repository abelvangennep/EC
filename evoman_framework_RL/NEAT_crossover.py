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
    counter  = 0
    while((len(superior_network) + len(inferior_network))-1 > counter):
        print(len(superior_network))
        print(superior_id_tracker)
        print(counter)
        if superior_network[superior_id_tracker].innov_id == inferior_network[inferior_id_tracker].innov_id:
            if superior_network[superior_id_tracker].enabled and inferior_network[inferior_id_tracker].enabled:
                child.append(Connection_Gene(superior_network[superior_id_tracker].inn, superior_network[superior_id_tracker].out, superior_network[superior_id_tracker].weight, superior_network[superior_id_tracker].innov_id, True))
            else:
                child.append(Connection_Gene(superior_network[superior_id_tracker].inn, superior_network[superior_id_tracker].out, superior_network[superior_id_tracker].weight, superior_network[superior_id_tracker].innov_id, False))
            
            inferior_id_tracker += 1
            superior_id_tracker += 1
            counter += 2

        elif superior_network[superior_id_tracker].innov_id > inferior_network[inferior_id_tracker].innov_id:
            child.append(Connection_Gene(superior_network[superior_id_tracker].inn, superior_network[superior_id_tracker].out, superior_network[superior_id_tracker].weight, superior_network[superior_id_tracker].innov_id, superior_network[superior_id_tracker].enabled))
            inferior_id_tracker += 1
            counter += 1

        elif superior_network[superior_id_tracker].innov_id < inferior_network[inferior_id_tracker].innov_id:
            child.append(Connection_Gene(inferior_network[superior_id_tracker].inn, inferior_network[superior_id_tracker].out, inferior_network[superior_id_tracker].weight, inferior_network[superior_id_tracker].innov_id, inferior_network[superior_id_tracker].enabled))
            superior_id_tracker += 1
            counter += 1

        else:
            print("if we get here something is going wrong in NEAT crossover")
            break

    return Individual(child)




