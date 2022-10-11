from pickle import TRUE
import random
import math
import numpy as np

class Individual():
    def __init__(self, sigma, network=[], fitness=None):
        """ Initialize individual for the NEAT population"""
        if network:
            self.network = network
        else:
            self.network = np.random.uniform(-1, 1, 265)
        self.sigma = sigma
        self.t = 1/math.sqrt(len(self.network))
        self.fitness = fitness

    def get_network(self):
        return self.network

    def set_network(self, network):
        self.network = network

    def set_sigma(self):
        self.sigma = self.sigma * math.exp(np.random.normal(0, self.t))
        if self.sigma < 0.0001:
            self.sigma = 0.0001
    
    def set_fitness(self, fitness):
        self.fitness = fitness
        
    def get_fitness(self):
        return self.fitness
        
    def print_network(self):
        for i in range(len(self.network)):
            print('From node ',self.network[i].get_inn().get_type(), ' ' , self.network[i].get_inn().get_id(), ' to node ', self.network[i].get_out().get_id(), ' ', self.network[i].get_out().get_type(), ' weight: ', self.network[i].get_weight(), ' innov_id: ', self.network[i].get_innov_id())



def calc_fitness_value(plife, elife,runtime):
    return 0.9*(100-elife)+0.1*plife-math.log(runtime)


