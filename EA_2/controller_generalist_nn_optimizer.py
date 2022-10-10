
"""
This file was used for parameter optimization.
"""
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
                  enemies=[1, 3, 4],
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
    num_iterations, number_generations, population_size, tournament_size, sigma = list_[0], list_[1], list_[2], \
                                                                                          list_[3], list_[4]

    overview = np.zeros((number_generations, 2))  # (maybe only for final)
    # Write a new initialize_network
    pop = [Individual(initialize_network(), sigma, i) for i in range(population_size)]
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

        #find best individual of population
        best_ind = pop[np.argmax(fitnesses)]
        enemy_win = []
        fitness_all_enemies = 0
        for enem in range(1,9):
            env2.update_parameter('enemies', [enem])
            f, p, e, t = env.play(pcont=best_ind)
            enemy_win.append(p>0)
            fitness_all_enemies+=f

        # Return the offspring
        offsprings_old = crossover(pop)
        offsprings = []
        for offspring in offsprings_old:
            offsprings.append(mutate(offspring))

        # Evaluate offsprings
        with concurrent.futures.ProcessPoolExecutor() as executor:
            fpet_off_results = executor.map(evaluate, offsprings)
        fpet_off = np.array([i for i in fpet_off_results])
        fitness_offspring = fpet_off[:, 0]
        # assign fitness to offsprings
        for i in range(len(offsprings)):
            offsprings[i].set_fitness(fitness_offspring[i])

        # Make some selection criterea to find a new population and return there corresponding fitness
        pop, fitnesses = select_population(pop, offsprings, tournament_size)

        # evaluate/run for whole new generation and assign fitness value
        max_score = np.argmax(fitnesses)
        mean = np.mean(fitnesses)
        std = np.std(fitnesses)
        print('gen: ', gen, '   Max fitness: ', max_score, '   mean fitness: ', mean, '   std fitness: ', std, ' won enemies: ', enemy_win, ' fitness 8 enemies sum: ', fitness_all_enemies)

        # keep track of solution improves if not do we want to do something????

    return fitness_all_enemies


# number_generations = 10
# population_size = 20


# if __name__ == '__main__':
#     neat_optimizer([2, number_generations, population_size, tournament_size, mutation_prob])

def neat_iterations_parallel(parameters):
    num_iterations = 2
    number_generations = 2
    population_size = 6
    sigma = parameters['sigma']
    tournament_size = parameters['tournament_size']

    print(parameters)

    best_fitnesses = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(neat_optimizer,
                               [[num_iterations, number_generations, population_size, tournament_size, sigma]
                                for _ in range(num_iterations)])

        res = [i for i in results]

    dict = {'loss': -np.mean(res), 'status': STATUS_OK, 'eval_time': time.time(), 'loss_variance': np.var(res)}
    print(dict)
    return -np.mean(res)


if __name__ == '__main__':
    space = hp.choice('Type_of_model', [{
        'sigma': hp.uniform("sigma", .0001, 1),
        'tournament_size': hp.quniform("tournament_size", 2, 10, 1),

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
