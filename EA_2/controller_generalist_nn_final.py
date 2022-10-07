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
from NN_EA_controller import player_controller
from NN_EA import initialize_network, Individual
from NN_EA_selection import select_population
from NN_EA_crossover import crossover
from NN_EA_mutate import mutate

# imports other libs
import numpy as np
import pandas as pd
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
# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
                  enemies=[1, 3, 4],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")


highest_species_id = 0


def simulation(env, p):
    f, p, e, t = env.play(pcont=p)
    return [f, p, e, t]


def evaluate(x):
    return simulation(env, x)



def run_neat(list_):
    """
    Run neat returns a 2 dimensional matrix,  inculding mean and max fitness
    """
    number_generations, population_size, tournament_size, mutation_prob, enemies = list_[0], list_[1], list_[2], \
                                                                                          list_[3],list_[4]

    overview = np.zeros((number_generations,2))

    # Update the enemy
    env.update_parameter('enemies', enemies)

    pop = [Individual(initialize_network(), i) for i in range(population_size)]
    best_inviduals = []

    for gen in range(number_generations): #number of generations
        start_gen = time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            fpet_pop_results = executor.map(evaluate, pop)  # fpet = fitness, player life, enemy life, time
        fpet_pop = np.array([i for i in fpet_pop_results])
        # assign fitnesses to inds
        fitnesses = list(fpet_pop[:, 0])
        enemy_life = fpet_pop[:, 2]
        for i in range(len(pop)):
            pop[i].set_fitness(fitnesses[i])

        overview[gen,0] = sum(fitnesses)/len(fitnesses)
        overview[gen,1] = max(fitnesses)

        good_individual = pop[fitnesses.index(max(fitnesses))]
        best_inviduals.append((max(fitnesses),good_individual))
        # Return the offspring
        offspring = crossover(pop)
        offsprings = []
        for o in offspring:
            offsprings.append(mutate(o, mutation_prob))

        # Evaluate offsprings
        with concurrent.futures.ProcessPoolExecutor() as executor:
            fpet_off_results = executor.map(evaluate, offsprings)
        fpet_off = np.array([i for i in fpet_off_results])
        fitness_offspring = fpet_off[:, 0]
        # assign fitness to offsprings
        for i in range(len(offspring)):
            offsprings[i].set_fitness(fitness_offspring[i])

        # Make some selection criterea to find a new population and return there corresponding fitness
        pop, fitnesses = select_population(pop, offsprings, tournament_size)


        #evaluate/run for whole new generation and assign fitness value
        print('Generation ', gen, ' took ', time.time()-start_gen, ' seconds to elapse. Highest fitness value was ', max(fitnesses), 'lowest enemy life: ',min(enemy_life) )

    best_fitness = max(best_inviduals,key=lambda item:item[0])
    vfitness, vplayerlife, venemylife, vtime = env.play(best_fitness[1])
    best_ind_gain = vplayerlife-venemylife
    return overview, best_ind_gain



def final_experiment_data(runs = 10, number_generations = 20, population_size = 45, tournament_size = 3, mutation_prob = 0.3, enemies=[1,3,4]):
    "Writes the best outcomes to a seperate csv-file"
    plot_max_fit = np.zeros((number_generations,runs))
    plot_mean_fit = np.zeros((number_generations,runs))
    scores_of_best_individuals = []
    for i in range(int(runs/2)):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(run_neat, [[number_generations, population_size, tournament_size, mutation_prob, enemies] for _ in range(2)])

        print('Finished ', 2*i, ' runs out of ', runs)

        for index, new_cols in enumerate(results):
            overview = new_cols[0]
            best_ind_gain= new_cols[1]
            scores_of_best_individuals.append(best_ind_gain)
            plot_mean_fit[:,i*2+index] = overview[:,0]
            plot_max_fit[:,i*2+index] = overview[:,1]

    print(scores_of_best_individuals)
    df_boxplot = pd.DataFrame(scores_of_best_individuals)
    df_boxplot.to_csv('boxplot_EA2'+str(runs)+'runs_enemy'+str(enemies[0])+'.csv', index_label=None)



    df_max_fit = pd.DataFrame(plot_max_fit)
    df_max_fit.to_csv('max_fitness_EA2'+str(runs)+'runs_enemy'+str(enemies[0])+'.csv', index_label=None)
    df_mean_fit = pd.DataFrame(plot_mean_fit)
    df_mean_fit.to_csv('mean_fitness_EA2'+str(runs)+'runs_enemy'+str(enemies[0])+'.csv', index_label=None)

if __name__ == '__main__':
    final_experiment_data(runs = 2, number_generations = 5, population_size = 20,  tournament_size = 3, mutation_prob = 0.3, enemies=[1,3,4]) #runs has to be even number

