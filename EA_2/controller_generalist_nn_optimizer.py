
"""
This file was used for parameter optimization.
"""
# imports framework
import os
import sys
import time
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

from demo_controller import player_controller
from NN_EA_selection import select_population
from NN_EA_crossover import crossover
from NN_EA_mutate import mutate

import numpy as np
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import concurrent.futures

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'controller_specialist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

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
    return np.array(list(map(lambda y: simulation(env, y), x)))


def neat_optimizer(list_):
    num_iterations, number_generations, population_size, tournament_size, sigma = list_[0], list_[1], list_[2], \
                                                                                          list_[3], list_[4]
    # Write a new initialize_network
    
    pop = np.random.uniform(-1, 1, (population_size,265))
    new_column = np.full(shape=(population_size,1), fill_value=sigma,dtype=np.float)
    
    pop = np.append(pop, new_column, axis=1)


    for gen in range(number_generations):  # number of generations
        # Evaluate population
        fpet_pop = evaluate(pop[:,0:265])  # fpet = fitness, player life, enemy life, time
        # assign fitnesses to inds
        fitnesses = fpet_pop[:, 0]

        # find best individual of population
        best_ind = pop[np.argmax(fitnesses)]
        enemy_win = []
        fitness_all_enemies = 0
        for enem in range(1,9):
            env2.update_parameter('enemies', [enem])
            f, p, e, t = env2.play(pcont=best_ind[0:265])
            enemy_win.append(e==0)
            fitness_all_enemies+=f

        # Return the offspring
        new_pop = crossover(pop, population_size)
        
        # Evaluate offsprings
        fpet_new = evaluate(new_pop[:,0:265])
        fitness_new = fpet_new[:, 0]
        
        # Make some selection criterea to find a new population and return there corresponding fitness
        pop, fitnesses = select_population(new_pop, fitness_new, tournament_size, population_size)

        #  check if variation is below treshold
        pop[pop[:,265] < sigma, 265] = 0.0001

        # evaluate/run for whole new generation and assign fitness value
        max_score = np.max(fitnesses)
        mean = np.mean(fitnesses)
        std = np.std(fitnesses)
        print('gen: ', gen, '   Max fitness: ', max_score, '   mean fitness: ', mean, '   std fitness: ', std, ' won enemies: ', enemy_win, ' fitness 8 enemies sum: ', fitness_all_enemies)

        # keep track of solution improves if not do we want to do something????

    return fitness_all_enemies/8


def neat_iterations_parallel(parameters):
    num_iterations = 3
    number_generations = 10
    population_size = 60
    sigma = parameters['sigma']
    tournament_size = int(parameters['tournament_size'])

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
        'sigma': hp.choice("sigma", [0.5, .1,.05,.01,.005,.001]),
        'tournament_size': hp.quniform("tournament_size", 2, 5, 1),

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
