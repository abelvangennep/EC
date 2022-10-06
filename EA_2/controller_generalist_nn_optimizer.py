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

sys.path.insert(0, 'evoman')
from environment import Environment
from NEAT_controller import player_controller
# from demo_controller import player_controller
from NEAT import Node_Gene, Connection_Gene, initialize_network, Individual, calc_fitness_value
from NEAT_selection import select_population
from NEAT_crossover import crossover
from NEAT_mutate import mutate
import numpy as np
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
                  enemies=[7, 8],
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


def neat_optimizer(list_):
    num_iterations, number_generations, population_size, tournament_size, mutation_prob = list_[0], list_[1], list_[2], \
                                                                                          list_[3], list_[4]

    overview = np.zeros((number_generations, 2))  # (maybe only for final)
    # Write a new initialize_network
    pop = [Individual(initialize_network(), i) for i in range(population_size)]
    best_three_gens = 0
    for gen in range(number_generations):  # number of generations

        # Evaluate population
        with concurrent.futures.ProcessPoolExecutor() as executor:
            fpet_pop_results = executor.map(evaluate, pop)  # fpet = fitness, player life, enemy life, time
        fpet_pop = np.array([i for i in fpet_pop_results])
        # assign fitnesses to inds
        fitnesses = fpet_pop[:, 0]
        for i in range(len(pop)):
            pop[i].set_fitness(fitnesses[i])

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

        # evaluate/run for whole new generation and assign fitness value
        max_score = np.argmax(fitnesses)
        mean = np.mean(fitnesses)
        std = np.std(fitnesses)
        print('gen: ', gen, '   Max fitness: ', max_score, '   mean fitness: ', mean, '   std fitness: ', std)

        # keep track of solution improves if not do we want to do something????

        if gen >= number_generations - num_iterations:
            best_three_gens += max_score
    return best_three_gens / 3


# number_generations = 10
# population_size = 20


# if __name__ == '__main__':
#     neat_optimizer([2, number_generations, population_size, tournament_size, mutation_prob])

def neat_iterations_parallel(parameters):
    num_iterations = 2
    number_generations = 2
    population_size = 6
    # tournament_size = 4
    mutation_prob = parameters['mutation_prob']
    tournament_size = parameters['tournament_size']

    print(parameters)

    best_fitnesses = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(neat_optimizer,
                               [[num_iterations, number_generations, population_size, tournament_size, mutation_prob]
                                for _ in range(num_iterations)])

        res = [i for i in results]

    dict = {'loss': -np.mean(res), 'status': STATUS_OK, 'eval_time': time.time(), 'loss_variance': np.var(res)}
    print(dict)
    return -np.mean(res)


if __name__ == '__main__':
    space = hp.choice('Type_of_model', [{
        'mutation_prob': hp.uniform("mutation_prob", .2, .7),
        'tournament_size': hp.quniform("tournament_size", 2, 5,1),

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
