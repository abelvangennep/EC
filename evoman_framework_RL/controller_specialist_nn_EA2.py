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
from neat_controller import player_controller
# from demo_controller import player_controller
from Neat import Node_Gene, Connection_Gene, initialize_network, Individual, calc_fitness_value
# from neat_selection import parent_selection
from neat_selection_EA2 import parent_selection
from Neat_speciation_EA2 import speciation, calc_avg_dist, Species
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
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  speed="fastest",
                  enemymode="static",
                  level=2)

# tests saved demo solutions for each enemy
# Parameters
# number_generations = 3
# population_size = 10
# mutation_prob =0
# compat_threshold =0
# link_insert_prob =0
# node_insert_prob =0
highest_species_id = 0


def run_neat(list_):
    number_generations, population_size, weight_mutation_lambda, compat_threshold, link_insertion_lambda, node_insertion_lambda, enemy = \
    list_[0], list_[1], list_[2], list_[3], list_[4], list_[5], list_[6]
    for en in enemy:
        # results = np.zeros(((number_generations+1)*population_size, 9)) #Generation, Individual, Parents, Species, Fitness, time, avg.gen fitness, avg.gen dist
        overview = np.zeros((number_generations, 2))
        # Update the enemy
        env.update_parameter('enemies', [en])
        # start with population, create 10 random individuals (1 for training now)
        pop = [Individual(initialize_network(), i) for i in range(population_size)]
        species = [Species(pop[0], 1, 0)]
        highest_species_id = 1
        highest_innov_id = 101
        id_node = 26
        for gen in range(number_generations):  # number of generations
            start_gen = time.time()
            #print('---- Starting with generation ', gen)
            fitnesses = []
            for pcont in pop:
                # print('Evaluating individual ', pcont.get_id())
                start_ind = time.time()
                vfitness, vplayerlife, venemylife, vtime = env.play(pcont)
                pcont.set_fitness(
                    calc_fitness_value(vplayerlife, venemylife, vtime) + 100)  # no negative fitness values
                fitnesses.append(calc_fitness_value(vplayerlife, venemylife, vtime))
                #print('Fitness value: ', calc_fitness_value(vplayerlife, venemylife, vtime), ' time elapsed: ',
                #     time.time() - start_ind)
                # results[gen * population_size+pcont.get_id(),0] = gen
                # results[gen * population_size + pcont.get_id(), 1] = pcont.get_id()
                # results[gen * population_size + pcont.get_id(), 5] = vfitness
                # results[gen * population_size + pcont.get_id(), 6] = time.time()-start_ind
            # overview[gen,0] = sum(fitnesses)/len(fitnesses)
            # overview[gen,1] = max(fitnesses)
            # results[gen*population_size,7] = sum(fitnesses)/len(fitnesses)
            # results[gen * population_size, 8] = calc_avg_dist(pop)
            species, highest_species_id = speciation(pop, species, highest_species_id, compat_threshold)  # The speciation function takes whole population as list of individuals and returns # a list of lists with individuals [[1,2], [4,5,8], [3,6,9,10], [7]] for example with 10 individuals
            for specie in species:
                specie.print()
            # print(species)
            # add species information to individual
            # for m in range(len(pop_grouped)):
            # for j in range(len(pop_grouped[m])):
            # results[gen*population_size+pop_grouped[m][j].get_id(), 3] = pop_grouped[m][j].get_species()
            # print('There are ', len(species), ' different species now in total')
            parents = parent_selection(species)  # This function returns pairs of parents which will be mated. In total the number of pairs equal to the number of offsprings we want to generate
            children = []
            for temp, pair in enumerate(parents):
                children.append(crossover(pair[0], pair[1]))  # for loop needed to cross each pair of parents
            # write parents in result table
            # results[(gen+1)*population_size+temp,2] = gen*100+min(pair[0].get_id(),pair[1].get_id())*10+max(pair[0].get_id(),pair[1].get_id())

            for m in range(len(children)):
                children[m], id_node, highest_innov_id, string = mutate(children[m], id_node, highest_innov_id,
                                                                        weight_mutation_lambda, link_insertion_lambda,
                                                                        node_insertion_lambda)
                children[m].set_id(m)
                # results[(gen + 1) * population_size + m, 4] = string

            # evaluate/run for whole new generation and assign fitness value
            pop = children
            # print('Generation ', gen, ' took ', time.time()-start_gen, ' seconds to elapse. Highest fitness value was ', max(fitnesses) )

        # results_df = pd.DataFrame(results, columns = ['Generation', 'Individual', 'Parents', 'Species', 'Mutation', 'Fitness', 'Time elapsed', 'Avg. fitness', 'Avg. distance'])
        # results_df.to_csv('test2.csv')
        # print(results_df)
        return overview


number_generations = 15
population_size = 45
compat_threshold = 4.3
weight_mutation_lambda = 0.6
link_insertion_lambda = 0.34
node_insertion_lambda = .12
enemy = [4]


# run_neat([number_generations, population_size, compat_threshold, weight_mutation_lambda, link_insertion_lambda, node_insertion_lambda, enemy])
def final_experiment_data(runs=10, number_generations=20, population_size=45, compat_threshold=4.3,
                          weight_mutation_lambda=0.6, link_insertion_lambda=0.34, node_insertion_lambda=.12, enemy=[4]):
    plot_max_fit = np.zeros((number_generations, runs))
    plot_mean_fit = np.zeros((number_generations, runs))
    for i in range(int(runs / 2)):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(run_neat,
                                   [[number_generations, population_size, weight_mutation_lambda, compat_threshold,
                                     link_insertion_lambda, node_insertion_lambda, enemy] for _ in range(2)])
        # new_cols = run_neat(number_generations, population_size, compat_threshold, weight_mutation_lambda, link_insertion_lambda, node_insertion_lambda, enemy)
        print('Finished ', 2 * i, ' runs out of ', runs)
        j = 0
        for new_cols in results:
            plot_mean_fit[:, i * 2 + j] = new_cols[:, 0]
            plot_max_fit[:, i * 2 + j] = new_cols[:, 1]
            j += 1
    df_max_fit = pd.DataFrame(plot_max_fit)
    df_max_fit.to_csv('max_fitness_' + str(runs) + 'runs_enemy' + str(enemy[0]) + '.csv', index_label=None)
    df_mean_fit = pd.DataFrame(plot_mean_fit)
    df_mean_fit.to_csv('mean_fitness_' + str(runs) + 'runs_enemy' + str(enemy[0]) + '.csv', index_label=None)


def final_experiment_plot(max_fit_csv, mean_fit_csv):
    plot_max_fit = np.loadtxt(max_fit_csv, delimiter=',', skiprows=1)[:, 1:]
    print(plot_max_fit)
    plot_mean_fit = np.loadtxt(mean_fit_csv, delimiter=',', skiprows=1)[:, 1:]
    number_generations = len(plot_max_fit)
    x = [i for i in range(1, number_generations + 1)]
    data_box = [plot_max_fit[i, :] for i in range(number_generations)]
    y1 = np.average(plot_mean_fit[:], axis=1)
    y2 = np.average(plot_max_fit[:], axis=1)
    plt.plot(x, y1, 'b-')
    plt.plot(x, y2, 'r-')
    plt.boxplot(data_box, labels=x)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.show()


# if __name__ == '__main__':
# final_experiment_data(runs = 10, number_generations = 20, population_size = 45, compat_threshold = 4.3, weight_mutation_lambda = 0.6, link_insertion_lambda=0.34, node_insertion_lambda=.12, enemy=[4]) #runs has to be even number
# final_experiment_plot('max_fitness_10runs_enemy4.csv', 'mean_fitness_10runs_enemy4.csv')


def neat_optimizer(list_):
    number_generations, population_size, weight_mutation_lambda, compat_threshold, link_insertion_lambda, node_insertion_lambda, enemy = \
    list_[0], list_[1], list_[2], list_[3], list_[4], list_[5], list_[6]

    for en in enemy:
        overview = np.zeros((number_generations, 2))
        # Update the enemy
        env.update_parameter('enemies', [en])
        # start with population, create 10 random individuals (1 for training now)
        pop = [Individual(initialize_network(), i) for i in range(population_size)]
        species = [Species(pop[0], 1, 0)]
        highest_species_id = 1
        highest_innov_id = 101
        id_node = 26
        best_three_gens = 0
        for gen in range(number_generations):  # number of generations
            fitnesses = []
            venemylifes = []
            for pcont in pop:
                vfitness, vplayerlife, venemylife, vtime = env.play(pcont)
                venemylifes.append(venemylife)
                pcont.set_fitness(
                    calc_fitness_value(vplayerlife, venemylife, vtime) + 100)  # no negative fitness values
                fitnesses.append(calc_fitness_value(vplayerlife, venemylife, vtime))
            overview[gen, 0] = sum(fitnesses) / len(fitnesses)
            overview[gen, 1] = calc_avg_dist(pop)
            #print('Num individuals going in speciation: ', len(pop))
            species, highest_species_id = speciation(pop, species, highest_species_id, compat_threshold)  # The speciation function takes whole population as list of individuals and returns # a list of lists with individuals [[1,2], [4,5,8], [3,6,9,10], [7]] for example with 10 individuals
            # add species information to individual
            parents = parent_selection(species)  # This function returns pairs of parents which will be mated. In total the number of pairs equal to the number of offsprings we want to generate
            #print('Number of parents: ', len(parents))
            children = []

            for temp, pair in enumerate(parents):
                children.append(crossover(pair[0], pair[1]))  # for loop needed to cross each pair of parents
            #print('Number of children: ', len(children))
            for m in range(len(children)):
                children[m], id_node, highest_innov_id, string = mutate(children[m], id_node, highest_innov_id,
                                                                        weight_mutation_lambda, link_insertion_lambda,
                                                                        node_insertion_lambda)
                children[m].set_id(m)

            # evaluate/run for whole new generation and assign fitness value
            pop = children
            print('Individuals in next generation: ', len(pop))
            max_value = max(fitnesses)
            print('gen: ', gen, '   fitness: ', max_value, "    venemylife:", min(venemylifes))
            if gen >= number_generations - 3:
                best_three_gens += max_value

    return best_three_gens / 3


neat_optimizer([number_generations, population_size, weight_mutation_lambda, compat_threshold,  link_insertion_lambda,
                node_insertion_lambda, enemy])


def neat_iterations(parameters):
    num_iterations = 1
    number_generations = 2
    population_size = int(parameters['population_size'])
    weight_mutation_lambda = parameters['weight_mutation_lambda']
    compat_threshold = parameters['compat_threshold']
    link_insertion_lambda = parameters['link_insertion_lambda']
    node_insertion_lambda = parameters['node_insertion_lambda']
    enemy = [4]

    print(parameters)

    best_fitnesses = []

    for iteration in range(num_iterations):
        print(iteration)
        best_fitnesses.append(
            neat_optimizer(number_generations, population_size, weight_mutation_lambda, compat_threshold,
                           link_insertion_lambda, node_insertion_lambda, enemy))

    return {'loss': -np.mean(best_fitnesses),
            'status': STATUS_OK,
            # -- store other results like this
            'eval_time': time.time(),
            'loss_variance': np.var(best_fitnesses)}


def neat_iterations_parallel(parameters):
    num_iterations = 4
    number_generations = 10
    population_size = int(parameters['population_size'])
    weight_mutation_lambda = parameters['weight_mutation_lambda']
    compat_threshold = parameters['compat_threshold']
    link_insertion_lambda = parameters['link_insertion_lambda']
    node_insertion_lambda = parameters['node_insertion_lambda']
    enemy = [4]

    print(parameters)

    best_fitnesses = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(neat_optimizer,
                               [[number_generations, population_size, weight_mutation_lambda, compat_threshold,
                                 link_insertion_lambda, node_insertion_lambda, enemy] for _ in range(num_iterations)])

        res = [i for i in results]

    dict = {'loss': -np.mean(res), 'status': STATUS_OK, 'eval_time': time.time(), 'loss_variance': np.var(res)}
    print(dict)
    return -np.mean(res)


'''
if __name__ == '__main__':

    space = hp.choice('Type_of_model',[{
            'population_size': hp.quniform("population_size", 10, 100, 1),
            'weight_mutation_lambda': hp.uniform("weight_mutation_lambda", .5, 3),
            'compat_threshold': hp.uniform("compat_threshold", 2, 15),
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
'''