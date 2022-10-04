import os
import sys
import time


sys.path.insert(0, 'evoman')
from environment import Environment
from neat_controller import player_controller
# from demo_controller import player_controller
from Neat import Node_Gene, Connection_Gene, initialize_network, Individual, calc_fitness_value
from neat_selection import parent_selection
from Neat_speciation import speciation, calc_avg_dist
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



def neat_optimizer(list_):
    number_generations, population_size, weight_mutation_lambda, compat_threshold, link_insert_prob, node_insert_prob, enemy = list_[0], list_[1], list_[2], list_[3], list_[4], list_[5], list_[6]
    for en in enemy:
        overview = np.zeros((number_generations, 2))
        # Update the enemy
        env.update_parameter('enemies', [en])
        # start with population, create 10 random individuals (1 for training now)
        pop = [Individual(initialize_network(), i) for i in range(population_size)]
        highest_innov_id = 101
        id_node = 26
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

            species = speciation(pop,
                                 compat_threshold)  # The speciation function takes whole population as list of individuals and returns # a list of lists with individuals [[1,2], [4,5,8], [3,6,9,10], [7]] for example with 10 individuals

            # add species information to individual
            parents = parent_selection(
                species)  # This function returns pairs of parents which will be mated. In total the number of pairs equal to the number of offsprings we want to generate
            children = []

            for temp, pair in enumerate(parents):
                children.append(crossover(pair[0], pair[1]))  # for loop needed to cross each pair of parents
                # write parents in result table
                temp += 1
            for m in range(len(children)):
                children[m], id_node, highest_innov_id, string = mutate(children[m], id_node, highest_innov_id,
                                                                        weight_mutation_lambda, link_insert_prob,
                                                                        node_insert_prob)
                children[m].set_id(m)

            # evaluate/run for whole new generation and assign fitness value
            pop = children
            max_value = max(fitnesses)
            print('gen: ', gen, '   fitness: ', max_value, "    venemylife:", min(venemylifes))

    return max_value


def neat_iterations(parameters):
    num_iterations = 1
    number_generations = 10
    population_size = int(parameters['population_size'])
    weight_mutation_lambda = parameters['weight_mutation_lambda']
    compat_threshold = parameters['compat_threshold']
    link_insert_prob = parameters['link_insert_prob']
    node_insert_prob = parameters['node_insert_prob']
    enemy = [1]

    print(parameters)

    best_fitnesses = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(neat_optimizer,[[number_generations, population_size, weight_mutation_lambda, compat_threshold,
                           link_insert_prob, node_insert_prob, enemy] for _ in range(num_iterations)])
        for result in results:
            print(result)

    return {'loss': 0, #-np.mean(best_fitnesses),
            'status': STATUS_OK,
            # -- store other results like this
            'eval_time': time.time(),
            'loss_variance': 0} #np.var(best_fitnesses)}

if __name__ == '__main__':
    space = hp.choice('Type_of_model', [{
        'population_size': hp.quniform("population_size", 10, 11, 1),
        'weight_mutation_lambda': hp.uniform("weight_mutation_lambda", 0, 5),
        'compat_threshold': hp.uniform("compat_threshold", 1, 12),
        'link_insert_prob': hp.uniform("link_insert_prob", 0, 1),
        'node_insert_prob': hp.uniform("node_insert_prob", 0, 1),
    }])

    trials = Trials()
    best = fmin(
        neat_iterations,
        space,
        trials=trials,
        algo=tpe.suggest,
        max_evals=3,
    )

    print("The best combination of hyperparameters is:")
    print(best)