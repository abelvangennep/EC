#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import os
import sys
import time
import matplotlib.pyplot as plt

sys.path.insert(0, 'evoman')
from environment import Environment
from NEAT_controller import player_controller
# from demo_controller import player_controller
from NEAT import Node_Gene, Connection_Gene, initialize_network, Individual, calc_fitness_value
from NEAT_selection import parent_selection
from NEAT_speciation import speciation, calc_avg_dist, Species
from NEAT_crossover import crossover
from NEAT_mutate import mutate
# imports other libs
import numpy as np
import pandas as pd

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import concurrent.futures
import statistics

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'controller_specialist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 0

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  speed="fastest",
                  enemymode="static",
                  level=2)

highest_species_id = 0

bestindivids = []

def run_neat(list_):
    number_generations, population_size, weight_mutation_lambda, compat_threshold, link_insertion_lambda, node_insertion_lambda, enemy = list_[0], list_[1], list_[2], list_[3], list_[4], list_[5], list_[6]
    for en in enemy:
        overview = np.zeros((number_generations,2))
        # Update the enemy
        env.update_parameter('enemies', [en])
        #start with population
        pop = [Individual(initialize_network(), i) for i in range(population_size)]
        species = [Species(pop[0],1,0)]
        highest_species_id = 1
        highest_innov_id = 101
        id_node = 26
  
        for gen in range(number_generations): #number of generations
            start_gen = time.time()
            fitnesses = []
            bestind = pop[0]
            bestind.set_fitness(0)
            for pcont in pop:
                vfitness, vplayerlife, venemylife, vtime = env.play(pcont)
                pcont.set_fitness(calc_fitness_value(vplayerlife, venemylife, vtime)+100) # no negative fitness values
                if pcont.get_fitness() > bestind.get_fitness():
                    bestind = pcont
                fitnesses.append(calc_fitness_value(vplayerlife, venemylife, vtime))
            overview[gen,0] = sum(fitnesses)/len(fitnesses)
            overview[gen,1] = max(fitnesses)
            pop_grouped, species, highest_species_id = speciation(pop, species, highest_species_id, compat_threshold)#The speciation function takes whole population as list of individuals and returns # a list of lists with individuals [[1,2], [4,5,8], [3,6,9,10], [7]] for example with 10 individuals
            parents = parent_selection(pop_grouped) #This function returns pairs of parents which will be mated. In total the number of pairs equal to the number of offsprings we want to generate

            children = []
            for temp, pair in enumerate(parents):
                children.append(crossover(pair[0], pair[1])) #for loop needed to cross each pair of parents

            for m in range(len(children)):
                children[m], id_node, highest_innov_id, string = mutate(children[m], id_node, highest_innov_id, weight_mutation_lambda, link_insertion_lambda, node_insertion_lambda)
                children[m].set_id(m)
            pop = children
            print('Generation ', gen, ' took ', time.time()-start_gen, ' seconds to elapse. Highest fitness value was ', max(fitnesses) )
        bestindivids.append(bestind)
        for bestind in bestindivids:
            print('Best ind; ', bestind.get_id(), 'and its fitness: ', bestind.get_fitness())
        return overview



number_generations = 10
population_size = 45
compat_threshold = 4.3
weight_mutation_lambda = 0.6
link_insertion_lambda = 0.34
node_insertion_lambda = .12
enemy = [4]
if __name__ == '__main__':
    run_neat([number_generations, population_size, weight_mutation_lambda, compat_threshold, link_insertion_lambda, node_insertion_lambda, enemy])
    #final_experiment_data(runs = 2, number_generations = 5, population_size = 45, compat_threshold = 4.3, weight_mutation_lambda = 0.6, link_insertion_lambda=0.34, node_insertion_lambda=.12, enemy=[4]) #runs has to be even number
    #final_experiment_plot('max_fitness_10runs_enemy2.csv', 'mean_fitness_10runs_enemy2.csv')



