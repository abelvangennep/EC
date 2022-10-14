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

# To block print statements while itter
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

sys.path.insert(0, 'evoman')

from demo_controller import player_controller
from NN_EA_selection import select_population
from NN_EA_crossover import crossover


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
n_hidden_neurons = 10
enemies = [2,5,8]

# initializes environment for single objective mode (specialist)  with static enemy and ai player
# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
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
    return np.array(list(map(lambda y: simulation(env, y), x)))



def run_neat(list_):
    """
    Run neat returns a 2 dimensional matrix,  inculding mean and max fitness
    """
    number_generations, population_size, tournament_size, sigma = list_[0], list_[1], list_[2], list_[3]

    overview = np.zeros((number_generations,2))

    pop = np.random.uniform(-1, 1, (population_size,265))
    new_column = np.full(shape=(population_size,1), fill_value=sigma,dtype=np.float)
    
    pop = np.append(pop, new_column, axis=1)
    best_individuals  = []

    for gen in range(number_generations): #number of generations
        start_gen = time.time()

        fpet_pop = evaluate(pop[:,0:265])  # fpet = fitness, player life, enemy life, time

        # assign fitnesses to inds
        fitnesses = fpet_pop[:, 0]

        enemy_life = fpet_pop[:, 2]


        overview[gen,0] = sum(fitnesses)/len(fitnesses)
        overview[gen,1] = max(fitnesses)

        good_individual = pop[np.argmax(fitnesses)]
        best_individuals.append((max(fitnesses),good_individual))

        # Return the offspring
        new_pop = crossover(pop, population_size)

        # Evaluate offsprings
        fpet_new = evaluate(new_pop[:,0:265])
        fitness_new = fpet_new[:, 0]
       
        # Make some selection criterea to find a new population and return there corresponding fitness
        pop, fitnesses = select_population(new_pop, fitness_new, tournament_size, population_size)

        #evaluate/run for whole new generation and assign fitness value
        print('Generation ', gen, ' took ', time.time()-start_gen, ' seconds to elapse. Highest fitness value was ', np.max(fitnesses), 'lowest enemy life: ',np.min(enemy_life) )
    
    best_num_wins = 0
    best_fitness = -50
    best_individual = None
    best_ind_gains = []
    for good_individual in best_individuals:
        enemy_win = []
        fitness_all_enemies = 0
        gains = []
        for enem in range(1,9):
            env2.update_parameter('enemies', [enem])
            f, p, e, t = env2.play(pcont=good_individual[1][0:265])
            enemy_win.append(e==0)
            fitness_all_enemies+=f
            gains.append(p-e)

        if sum(enemy_win) > best_num_wins or (sum(enemy_win) == best_num_wins and fitness_all_enemies > best_fitness):

            best_individual = good_individual[1]
            #print(best_individual, ' shape: ', len(best_individual))
            best_num_wins = sum(enemy_win)
            best_fitness = good_individual[0]
            best_ind_gains = gains

        print('Test against all enemies; won enemies: ', enemy_win, 'Best num of wins', best_num_wins, ' fitness 8 enemies sum: ', fitness_all_enemies)
        
    return overview, best_fitness, best_individual, best_ind_gains



def final_experiment_data(runs = 10, number_generations = 20, population_size = 45, tournament_size = 3, mutation_prob = 0.3, enemies=[1,3,4], sigma=0.001):
    "Writes the best outcomes to a seperate csv-file"
    plot_max_fit = np.zeros((number_generations,runs))
    plot_mean_fit = np.zeros((number_generations,runs))
    best_ind_gains = np.zeros((8, runs))
    scores_of_best_individuals = []

    for i in range(int(runs/2)):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(run_neat, [[number_generations, population_size, tournament_size, sigma] for _ in range(2)])

        print('Finished ', 2*i, ' runs out of ', runs)

        for index, new_cols in enumerate(results):
            overview = new_cols[0]
            best_fitness= new_cols[1]
            #print('new_cols 2', new_cols[2][0:265], ' länge: ', len(new_cols[2][0:265]))
            np.savetxt('best_ind/'+str(index+2*i)+'.txt', new_cols[2][0:265])
            ind_gains = new_cols[3]
            scores_of_best_individuals.append(best_fitness)
            plot_mean_fit[:,i*2+index] = overview[:,0]
            plot_max_fit[:,i*2+index] = overview[:,1]
            best_ind_gains[:, i * 2 + index] = ind_gains
    
    print(scores_of_best_individuals)
    df_boxplot = pd.DataFrame(scores_of_best_individuals)
    df_boxplot.to_csv('boxplot_EA2'+str(runs)+'runs_enemy'+str(enemies[0])+'.csv', index_label=None)



    df_max_fit = pd.DataFrame(plot_max_fit)
    df_max_fit.to_csv('max_fitness_EA2'+str(runs)+'runs_enemy'+str(enemies[0])+'.csv', index_label=None)
    df_mean_fit = pd.DataFrame(plot_mean_fit)
    df_mean_fit.to_csv('mean_fitness_EA2'+str(runs)+'runs_enemy'+str(enemies[0])+'.csv', index_label=None)

    df_ind_gains = pd.DataFrame(best_ind_gains)
    df_ind_gains.to_csv('best_individual_gains_EA2'+str(runs)+'runs_enemy_group'+str(enemies[0])+'.csv', index_label=None)

if __name__ == '__main__':
    final_experiment_data(runs = 4, number_generations = 1, population_size = 20,  tournament_size = 3, mutation_prob = 0.3, enemies=enemies, sigma=0.1) #runs has to be even number

