import random

class Node_Gene():
    """ Saves node genes"""
    def __init__(self, type, id):
        self.type = type #Input, Hidden, Output
        self.id = id
    def get_type(self):
        return self.type
    def get_id(self):
        return self.id


class Connection_Gene():
    """ Saves connection genes"""
    def __init__(self, in_node, out_node, weight, innov_id, enabled):
        self.inn = in_node
        self.out = out_node
        self.weight = weight
        self.innov_id = innov_id
        self.enabled = enabled #Boolean True or False

    def get_inn(self):
        return self.inn
    def get_out(self):
        return self.out
    def get_innov_id(self):
        return self.innov_id
    def get_weight(self):
        return self.weight


class individual():
    def __init__(self, network):
        """ Initialize individual for the NEAT population"""
        self.network = network
        self.fitness = None

    def get_network(self):
        return self.network

    def get_fitness(self):
        return self.network

    def print_network(self):
        for i in range(len(self.network)):
            print('From node ',self.network[i].get_inn().get_type(), ' ' , self.network[i].get_inn().get_id(), ' to node ', self.network[i].get_out().get_id(), ' ', self.network[i].get_out().get_type(), ' weight: ', self.network[i].get_weight(), ' innov_id: ', self.network[i].get_innov_id())

def initialize_network():
    """ Initialize network with only 20 input nodes and 5 output nodes"""
    network = []
    for i in range(20):
        for j in range(5):
            network.append(Connection_Gene(Node_Gene('Input',i+1),Node_Gene('Output',21+j), random.uniform(-5,5),20*j+i+1, True))
    return network


net = initialize_network()
ind = individual(net)
ind.print_network()