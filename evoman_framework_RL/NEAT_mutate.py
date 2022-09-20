import random

def mutate(individual, weight_mutation_prob, link_insertion_prob, node_insertion_prob, id_node, highest_innov_id):
    if random.uniform(0, 1) < weight_mutation_prob:
        individual = adjust_weight(individual)

    if random.uniform(0, 1) < link_insertion_prob:
        individual =link_insertion(individual)

    if random.uniform(0, 1) < node_insertion_prob:
        individual = add_node(individual, id_node, highest_innov_id)

    return individual

def adjust_weight(individual):
    network = individual.get_network()
    single_connection = random.choice(network)
    single_connection.weight = random.uniform(-1,1)

    return individual

def add_node(individual, id_node, highest_innov_id):
    network = individual.get_network()
    single_connection = random.choice(network)
    single_connection.disable()
    individual.add_connection(single_connection.in_node, single_connection.out_node, single_connection.weight, id_node, highest_innov_id)
    
    return individual

def link_insertion(individual):
    return individual
