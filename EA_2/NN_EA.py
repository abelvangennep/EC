from pickle import TRUE
import random
import math

class Node_Gene():
    """ Saves node genes"""
    def __init__(self, type, id):
        self.type = type #Input, Hidden, Output
        self.value = None
        self.id = id

    def get_type(self):
        return self.type
        
    def get_value(self):
        return self.value

    def set_value(self,value):
        self.value = value

    def print_node(self):
        print('Type: ', self.type, ' ID: ', self.id, ' value: ', self.value)

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
        
    def disable(self):
        self.enabled = False
        return True
    def get_out(self):
        return self.out
    def get_innov_id(self):
        return self.innov_id
    def get_weight(self):
        return self.weight
    def set_weight(self, w):
        self.weight = w


class Individual():
    def __init__(self, network, id = None, sp = 0):
        """ Initialize individual for the NEAT population"""
        self.network = network
        self.highest_innov = 0
        self.fitness = None
        self.id = id
        self.species = sp

    def get_network(self):
        return self.network

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
    
    def add_connection(self, in_node, out_node, weight, id_node, highest_innov_id):
        hidden_node = Node_Gene("Hidden", id_node)
        self.network.append(Connection_Gene(in_node, hidden_node, weight, highest_innov_id, True))
        self.network.append(Connection_Gene(hidden_node, out_node, 1, highest_innov_id+1, True))
        return True
        
    def print_network(self):
        for i in range(len(self.network)):
            print('From node ',self.network[i].get_inn().get_type(), ' ' , self.network[i].get_inn().get_id(), ' to node ', self.network[i].get_out().get_id(), ' ', self.network[i].get_out().get_type(), ' weight: ', self.network[i].get_weight(), ' innov_id: ', self.network[i].get_innov_id())

    def set_species(self, sp):
        self.species = sp

    def get_species(self):
        return self.species



def initialize_network():
    """ Initialize network with only 20 input nodes and 5 output nodes"""
    network = []
    for h in range(10):
        for j in range(20):
            network.append(
                Connection_Gene(Node_Gene('Input', j + 1), Node_Gene('Hidden', 31 + h), random.uniform(-1, 1), 20 * h + j + 1, True))
    for b in range(10):
        network.append(
            Connection_Gene(Node_Gene('Bias', 21 + b), Node_Gene('Hidden', 31 + b), random.uniform(-1, 1), 201 + b , True))
    for i in range(5):
        for h in range(10):
            network.append(
                Connection_Gene(Node_Gene('Hidden', 21 + h), Node_Gene('Output', 46 + i), random.uniform(-1, 1), 211 + 10 * i + h, True))
    for i in range(5):
        network.append(
            Connection_Gene(Node_Gene('Bias', 41 + i), Node_Gene('Output', 46 + i), random.uniform(-1, 1), 261+i, True))
    return network

def calc_fitness_value(plife, elife,runtime):
    return 0.9*(100-elife)+0.1*plife-math.log(runtime)
#net1 = initialize_network()
#net2 = initialize_network()
#net3 = initialize_network()
#net4 = initialize_network()
#net5 = initialize_network()

