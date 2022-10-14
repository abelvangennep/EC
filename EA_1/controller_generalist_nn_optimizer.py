#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################
''' This file was used for parameter optimization.'''
# imports framework
import os
import sys
import time
import matplotlib.pyplot as plt
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


sys.path.insert(0, 'evoman')
with HiddenPrints():
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
                  enemies=[2,5,8],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

env2 = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  speed="fastest",
                  enemymode="static",
                  level=2)


highest_species_id = 0

def simulation(env, p):
    f, p, e, t = env.play(pcont=p)
    return [f, p, e, t]


def evaluate(x):
    return simulation(env, x)

def neat_optimizer(list_):
    num_iterations, number_generations, population_size, weight_mutation_lambda, compat_threshold, link_insertion_lambda, node_insertion_lambda = list_[0], list_[1], list_[2], list_[3], list_[4], list_[5], list_[6]

    overview = np.zeros((number_generations,2))
    pop = [Individual(initialize_network(), i) for i in range(population_size)]
    species = [Species(pop[0], 1)]
    highest_species_id = 1
    highest_innov_id = 101
    id_node = 26
    best_three_gens = 0
    for gen in range(number_generations): #number of generations
        # Get fitness according to original fitness score
        #fitnesses = np.array(list(map(lambda y: env.play(pcont=y)[0], pop)))
        # Evaluate population
        with concurrent.futures.ProcessPoolExecutor() as executor:
            fpet_pop_results = executor.map(evaluate, pop)  # fpet = fitness, player life, enemy life, time
        fpet_pop = np.array([i for i in fpet_pop_results])
        # assign fitnesses to inds
        fitnesses = fpet_pop[:, 0]
        for i in range(len(pop)):
            pop[i].set_fitness(fitnesses[i])

        #find best individual of population
        best_ind = pop[np.argmax(fitnesses)]
        enemy_win = []
        fitness_all_enemies = 0
        for enem in range(1,9):
            env2.update_parameter('enemies', [enem])
            f, p, e, t = env2.play(pcont=best_ind)
            enemy_win.append(e==0)
            fitness_all_enemies+=f

        #solutions = [pop, fitnesses]
        #env.update_solutions(solutions)

        pop_grouped, species, highest_species_id = speciation(pop, species, highest_species_id, compat_threshold) #The speciation function takes whole population as list of individuals and returns # a list of lists with individuals [[1,2], [4,5,8], [3,6,9,10], [7]] for example with 10 individuals
        
        #add species information to individual
        parents = parent_selection(pop_grouped) #This function returns pairs of parents which will be mated. In total the number of pairs equal to the number of offsprings we want to generate
        children = []

        for temp, pair in enumerate(parents):
            children.append(crossover(pair[0], pair[1])) #for loop needed to cross each pair of parents

        for m in range(len(children)):
            children[m], id_node, highest_innov_id, string = mutate(children[m], id_node, highest_innov_id,weight_mutation_lambda, link_insertion_lambda, node_insertion_lambda)
            children[m].set_id(m)

        #evaluate/run for whole new generation and assign fitness value
        pop = children
        max_score = np.max(fitnesses)
        mean = np.mean(fitnesses)
        std = np.std(fitnesses)
        print('gen: ', gen, '   Max fitness: ', max_score, '   mean fitness: ', mean, '   std fitness: ', std, ' won enemies: ', enemy_win, ' fitness 8 enemies sum: ', fitness_all_enemies)

    return fitness_all_enemies/8


def neat_iterations_parallel(parameters):
    num_iterations = 3
    number_generations = 10
    population_size = 60
    weight_mutation_lambda = parameters['weight_mutation_lambda']
    compat_threshold = parameters['compat_threshold']
    link_insertion_lambda = parameters['link_insertion_lambda']
    node_insertion_lambda = parameters['node_insertion_lambda']

    print(parameters)

    best_fitnesses = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(neat_optimizer,[[num_iterations, number_generations, population_size, weight_mutation_lambda, compat_threshold,
                           link_insertion_lambda, node_insertion_lambda] for _ in range(num_iterations)])

        res = [i for i in results]

    dict = {'loss': -np.mean(res),'status': STATUS_OK,'eval_time': time.time(),'loss_variance': np.var(res)}
    print(dict)
    return -np.mean(res)


if __name__ == '__main__':

    space = hp.choice('Type_of_model',[{
            #'population_size': hp.quniform("population_size", 10, 100, 1),
            'weight_mutation_lambda': hp.uniform("weight_mutation_lambda", .5, 3),
            'compat_threshold': hp.uniform("compat_threshold", 4, 15),
            'link_insertion_lambda': hp.uniform("link_insertion_lambda", 0.05, .5),
            'node_insertion_lambda': hp.uniform("node_insertion_lambda", 0.05, .5),
                }])


    trials = Trials()
    best = fmin(
        neat_iterations_parallel,
        space,
        trials=trials,
        algo=tpe.suggest,
        max_evals=25,
    )

    print("The best combination of hyperparameters is:")
    print(best)