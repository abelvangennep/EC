#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

''' This file is used to generate the final data for the plots in the assignment'''

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
                  enemies=[1, 3, 4],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")
def simulation(env, p):
    f, p, e, t = env.play(pcont=p)
    return [f, p, e, t]


def evaluate(x):
    return simulation(env, x)


highest_species_id = 0


def run_neat(list_):
    """
    Run neat returns a 2 dimensional matrix,  inculding mean and max fitness
    """
    number_generations, population_size, weight_mutation_lambda, compat_threshold, link_insertion_lambda, node_insertion_lambda, enemies = \
        list_[0], list_[1], list_[2], list_[3], list_[4], list_[5], list_[6]
    overview = np.zeros((number_generations, 2))
    # Update the enemy
    env.update_parameter('enemies', enemies)

    pop = [Individual(initialize_network(), i) for i in range(population_size)]
    species = [Species(pop[0],1,0)]
    highest_species_id = 1
    highest_innov_id = 101
    id_node = 26
    best_inviduals = []

    for gen in range(number_generations): #number of generations
        start_gen = time.time()
        fitnesses = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            fpet_pop_results = executor.map(evaluate, pop)  # fpet = fitness, player life, enemy life, time
        fpet_pop = np.array([i for i in fpet_pop_results])
        # assign fitnesses to inds
        fitnesses = list(fpet_pop[:, 0])
        enemy_life = fpet_pop[:,2]
        for i in range(len(pop)):
            pop[i].set_fitness(fitnesses[i]+100)

        #Store average and maximal fitness of generation
        overview[gen,0] = sum(fitnesses)/len(fitnesses)
        overview[gen,1] = max(fitnesses)

        #find best individual of generation
        good_individual = pop[fitnesses.index(max(fitnesses))]
        best_inviduals.append((max(fitnesses),good_individual))

        pop_grouped, species, highest_species_id = speciation(pop, species, highest_species_id, compat_threshold)#The speciation function takes whole population as list of individuals and returns # a list of lists with individuals [[1,2], [4,5,8], [3,6,9,10], [7]] for example with 10 individuals
        parents = parent_selection(pop_grouped) #This function returns pairs of parents which will be mated. In total the number of pairs equal to the number of offsprings we want to generate
        children = []

        for temp, pair in enumerate(parents):
            children.append(crossover(pair[0], pair[1])) #for loop needed to cross each pair of parents

        for m in range(len(children)):
            children[m], id_node, highest_innov_id, string = mutate(children[m], id_node, highest_innov_id, weight_mutation_lambda, link_insertion_lambda, node_insertion_lambda)
            children[m].set_id(m)

        #evaluate/run for whole new generation and assign fitness value
        pop = children
        print('Generation ', gen, ' took ', time.time()-start_gen, ' seconds to elapse. Highest fitness value was ', max(fitnesses), 'lowest enemy life: ',min(enemy_life) )

    best_fitness = max(best_inviduals,key=lambda item:item[0])
    vfitness, vplayerlife, venemylife, vtime = env.play(best_fitness[1])
    best_ind_gain = vplayerlife-venemylife
    return overview, best_ind_gain

#10 independent runs for each EA and each training group

def final_experiment_data(runs = 10, number_generations = 20, population_size = 45, compat_threshold = 4.3, weight_mutation_lambda = 0.6, link_insertion_lambda=0.34, node_insertion_lambda=.12, enemies = [1,3,4]):
    "Writes the best outcomes to a seperate csv-file"
    plot_max_fit = np.zeros((number_generations,runs))
    plot_mean_fit = np.zeros((number_generations,runs))
    scores_of_best_individuals = []
    for i in range(int(runs/2)):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(run_neat, [[number_generations, population_size, weight_mutation_lambda, compat_threshold,
                           link_insertion_lambda, node_insertion_lambda, enemies] for _ in range(2)])

        print('Finished ', 2*i+2, ' runs out of ', runs)
        #results = [j for j in future]
        #print('länge: ', type(results), ' arrray: ', results)
        for index, new_cols in enumerate(results):
            overview = new_cols[0]
            best_ind_gain= new_cols[1]
            scores_of_best_individuals.append(best_ind_gain)
            plot_mean_fit[:,i*2+index] = overview[:,0]
            plot_max_fit[:,i*2+index] = overview[:,1]

    print(scores_of_best_individuals)
    df_boxplot = pd.DataFrame(scores_of_best_individuals)
    df_boxplot.to_csv('boxplot_NEAT1'+str(runs)+'runs_enemy_group'+str(enemies[0])+'.csv', index_label=None)



    df_max_fit = pd.DataFrame(plot_max_fit)
    df_max_fit.to_csv('max_fitness_EA1'+str(runs)+'runs_enemy_group'+str(enemies[0])+'.csv', index_label=None)
    df_mean_fit = pd.DataFrame(plot_mean_fit)
    df_mean_fit.to_csv('mean_fitness_EA1'+str(runs)+'runs_enemy_group'+str(enemies[0])+'.csv', index_label=None)

if __name__ == '__main__':
    final_experiment_data(runs = 2, number_generations = 10, population_size = 20, compat_threshold = 8.9, weight_mutation_lambda = 2.4, link_insertion_lambda=0.22, node_insertion_lambda=.49, enemies = [1,3,4]) #runs has to be even number

