import random

def mutate(individual, id_node, highest_innov_id, weight_mutation_prob=0.9, link_insertion_prob=.05, node_insertion_prob=.05):
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

def link_insertion(individual, id_node, highest_innov_id):
    hidden_nodes = []
    all_nodes = {}
    for connection in individual.get_network():
        if connection.in_node.type == "Hidden":
            hidden_nodes.append(connection.in_node)

        if not connection.in_node == "Hidden":
            all_nodes.add(connection.in_node)
        if not connection.out_node == "Hidden": 
            all_nodes.add(connection.out_node)

    if hidden_nodes:
        selected_node = random.choice(hidden_nodes)
        all_nodes.remove(selected_node)
        connected_node = random.choice(all_nodes)
        weight = random.uniform(-1,1)
        if connected_node.type == "Input":
            individual.add_connection(connected_node, selected_node, weight, id_node, highest_innov_id)
        elif connected_node.type == "Output":
            individual.add_connection(selected_node, connected_node, weight, id_node, highest_innov_id)
        else:
            print("SHOULD NOT BE HERE, PROBLEM IN NEAT_MUTATION")

    return individual
