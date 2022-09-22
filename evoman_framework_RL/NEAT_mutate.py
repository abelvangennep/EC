import random
import numpy as np

def mutate(individual, id_node, highest_innov_id, weight_mutation_lambda = 3, link_insertion_prob=.05, node_insertion_prob=.05):
    mut_string = 0
    for i in range(np.random.poisson(weight_mutation_lambda, 1)[0]):
        individual = adjust_weight(individual)
        mut_string = mut_string+100

    if random.uniform(0, 1) < link_insertion_prob:
        individual =  link_insertion(individual, id_node, highest_innov_id)
        id_node +=1
        highest_innov_id += 1
        mut_string = mut_string + 20

    if random.uniform(0, 1) < node_insertion_prob:
        individual = add_node(individual, id_node, highest_innov_id)
        id_node +=1
        highest_innov_id += 2
        mut_string = mut_string + 3

    return individual, id_node, highest_innov_id, mut_string

def adjust_weight(individual):
    network = individual.get_network()
    for i in range(random.randint(1,10)):
        single_connection = random.choice(network)
        single_connection.weight = random.uniform(-1,1)

    return individual

def add_node(individual, id_node, highest_innov_id):
    network = individual.get_network()
    single_connection = random.choice(network)
    single_connection.disable()
    individual.add_connection(single_connection.inn, single_connection.out, single_connection.weight, id_node, highest_innov_id)
    
    return individual

def link_insertion(individual, id_node, highest_innov_id):
    hidden_nodes = []
    all_nodes = set()
    for connection in individual.get_network():
        if connection.inn.type == "Hidden":
            hidden_nodes.append(connection.inn)

        if not connection.inn == "Hidden":
            all_nodes.add(connection.inn)
        if not connection.out == "Hidden":
            all_nodes.add(connection.out)

    if hidden_nodes:
        selected_node = random.choice(hidden_nodes)
        all_nodes.remove(selected_node)
        connected_node = random.sample(all_nodes, 1)[0]
        weight = random.uniform(-1, 1)
        if connected_node.type == "Input":
            individual.add_connection(connected_node, selected_node, weight, id_node, highest_innov_id)
        elif connected_node.type == "Output":
            individual.add_connection(selected_node, connected_node, weight, id_node, highest_innov_id)
        else:
            print("SHOULD NOT BE HERE, PROBLEM IN NEAT_MUTATION")

    return individual
