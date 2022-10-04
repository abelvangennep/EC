###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import pickle
import sys
import csv

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

from map_enemy_id_to_name import id_to_name

# imports other libs
import time
import numpy as np
import os

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'individual_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.


class EvalEnvCallback:
    def __init__(
            self,
            eval_env,
            model_dir=None,
            raw_data_dir=None,
            n_eval_episodes: int = 5,
            eval_freq: int = 1,  # generations
    ):
        if raw_data_dir is not None and not os.path.exists(raw_data_dir):
            os.makedirs(raw_data_dir)
        if model_dir is not None and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.raw_data_dir = raw_data_dir
        self.model_dir = model_dir
        self.generations = 0

    def collect_data(self, best_genome, population):
        with open(f'{self.model_dir}/generation-{self.generations}', mode='wb') as model_file:
            pickle.dump(population, model_file)
        self.generations = self.generations + 1
        if self.generations % self.eval_freq == 0:
            wins = []
            rs = []
            ls = []
            for j in range(self.n_eval_episodes):
                fitness, plife, elife, time = self.eval_env.play(pcont=best_genome)

                rs.append(fitness)
                ls.append(time)
                if elife <= 0:
                    wins.append(1)
                else:
                    wins.append(0)
            with open(f'{self.raw_data_dir}/wins.csv', mode='a') as wins_file:
                wins_writer = csv.writer(wins_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
                wins_writer.writerow([self.generations, self.n_eval_episodes, ''] + wins)
            with open(f'{self.raw_data_dir}/fitness.csv', mode='a') as rewards_file:
                rewards_writer = csv.writer(rewards_file, delimiter=',', quotechar='\'',
                                            quoting=csv.QUOTE_NONNUMERIC)
                rewards_writer.writerow([self.generations, self.n_eval_episodes, ''] + rs)

            return [np.mean(rs), np.mean(ls)]

environments = [
    (
        n,
        [(
            Environment(
                enemies=[n],
                weight_player_hitpoint=weight_player_hitpoint,
                weight_enemy_hitpoint=1.0 - weight_player_hitpoint,
                playermode="ai",
                randomini='yes',
                  enemymode="static",
                logs='off',
                player_controller=player_controller(n_hidden_neurons),
                # show_display=True,
            ),
            Environment(
                enemies=[n],
                weight_player_hitpoint=1,
                weight_enemy_hitpoint=1,
                randomini='yes',
                playermode="ai",
                  enemymode="static",
                logs='off',
                player_controller=player_controller(n_hidden_neurons),
                # show_display=True,
            )
        ) for weight_player_hitpoint in [0.1, 0.4, 0.5, 0.6]]
    )
    for n in range(1, 9)
]


# default environment fitness is assumed for experiment

# env.state_to_log()  # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = 'train'  # train or test

# number of weights for multilayer with 10 hidden neurons

dom_u = 1
dom_l = -1
npop = 100
gens = 100
mutation = 0.2
last_best = 0
runs = 5


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


# normalizes
def norm(x, pfit_pop):
    if (max(pfit_pop) - min(pfit_pop)) > 0:
        x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


# tournament
def tournament(pop):
    c1 = np.random.randint(0, pop.shape[0], 1)
    c2 = np.random.randint(0, pop.shape[0], 1)

    if fit_pop[c1] > fit_pop[c2]:
        return pop[c1][0]
    else:
        return pop[c2][0]


# limits
def limits(x):
    if x > dom_u:
        return dom_u
    elif x < dom_l:
        return dom_l
    else:
        return x


# crossover
def crossover(pop):
    total_offspring = np.zeros((0, n_vars))

    for p in range(0, pop.shape[0], 2):
        p1 = tournament(pop)
        p2 = tournament(pop)

        n_offspring = np.random.randint(1, 3 + 1, 1)[0]
        offspring = np.zeros((n_offspring, n_vars))

        for f in range(0, n_offspring):

            cross_prop = np.random.uniform(0, 1)
            offspring[f] = p1 * cross_prop + p2 * (1 - cross_prop)

            # mutation
            for i in range(0, len(offspring[f])):
                if np.random.uniform(0, 1) <= mutation:
                    offspring[f][i] = offspring[f][i] + np.random.normal(0, 1)

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring


# kills the worst genomes, and replace with new best/random solutions
def doomsday(pop, fit_pop):
    worst = int(npop / 4)  # a quarter of the population
    order = np.argsort(fit_pop)
    orderasc = order[0:worst]

    for o in orderasc:
        for j in range(0, n_vars):
            pro = np.random.uniform(0, 1)
            if np.random.uniform(0, 1) <= pro:
                pop[o][j] = np.random.uniform(dom_l, dom_u)  # random dna, uniform dist.
            else:
                pop[o][j] = pop[order[-1:]][0][j]  # dna from best

        fit_pop[o] = evaluate([pop[o]])

    return pop, fit_pop


# loads file with the best solution for testing
# if run_mode == 'test':
#     bsol = np.loadtxt(experiment_name + '/best.txt')
#     print('\n RUNNING SAVED BEST SOLUTION \n')
#     env.update_parameter('speed', 'normal')
#     evaluate([bsol])
#
#     sys.exit(0)

# initializes population loading old solutions or generating new ones

# else:
#
#     print('\nCONTINUING EVOLUTION\n')
#
#     env.load_state()
#     pop = env.solutions[0]
#     fit_pop = env.solutions[1]
#
#     best = np.argmax(fit_pop)
#     mean = np.mean(fit_pop)
#     std = np.std(fit_pop)
#
#     # finds last generation number
#     file_aux = open(experiment_name + '/gen.txt', 'r')
#     ini_g = int(file_aux.readline())
#     file_aux.close()

# saves results for first pop
# file_aux = open(experiment_name + '/results.txt', 'a')
# file_aux.write('\n\ngen best mean std')
# print('\n GENERATION ' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
#     round(std, 6)))
# file_aux.write(
#     '\n' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
# file_aux.close()

# evolution


for run in range(runs):
    print(f'Starting run {run}!')
    baseDir = f'FullTime/Base-GA-10/run{run}'

    if not os.path.exists(baseDir):
        os.makedirs(baseDir)

    for enemy_id, enemy_envs in environments:
        enemyDir = f'{baseDir}/{id_to_name(enemy_id)}'
        if not os.path.exists(enemyDir):
            os.makedirs(enemyDir)

        for env, eval_env in enemy_envs:
            n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
            pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
            print(np.array(pop).shape)
            fit_pop = evaluate(pop)
            best = np.argmax(fit_pop)
            mean = np.mean(fit_pop)
            std = np.std(fit_pop)
            ini_g = 0
            solutions = [pop, fit_pop]
            env.update_solutions(solutions)
            last_sol = fit_pop[best]

            modelDir = f'{enemyDir}/models/{({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})}'
            # videoDir = f'{enemyDir}/videos/{({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})}'
            rawDataDir = f'{enemyDir}/raw-data/{({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})}'
            lengths_path = f'{enemyDir}/Evaluation_lengths.csv'
            rewards_path = f'{enemyDir}/Evaluation_rewards.csv'
            if not os.path.exists(modelDir):
                os.makedirs(modelDir)
            # if not os.path.exists(videoDir):
            #     os.makedirs(videoDir)
            if not os.path.exists(rawDataDir):
                os.makedirs(rawDataDir)

            evaluator = EvalEnvCallback(
                eval_env=eval_env,
                raw_data_dir=rawDataDir,
                model_dir=modelDir,
                n_eval_episodes=25,
            )

            notimproved = 0
            for i in range(ini_g + 1, gens):

                offspring = crossover(pop)  # crossover
                fit_offspring = evaluate(offspring)  # evaluation
                pop = np.vstack((pop, offspring))
                fit_pop = np.append(fit_pop, fit_offspring)

                best = np.argmax(fit_pop)  # best solution in generation
                fit_pop[best] = float(evaluate(np.array([pop[best]]))[0])  # repeats best eval, for stability issues
                best_sol = fit_pop[best]

                evaluator.collect_data(pop[best], fit_pop)

                # selection
                fit_pop_cp = fit_pop
                fit_pop_norm = np.array(list(map(lambda y: norm(y, fit_pop_cp),
                                                 fit_pop)))  # avoiding negative probabilities, as fitness is ranges from negative numbers
                probs = (fit_pop_norm) / (fit_pop_norm).sum()
                chosen = np.random.choice(pop.shape[0], npop, p=probs, replace=False)
                chosen = np.append(chosen[1:], best)
                pop = pop[chosen]
                fit_pop = fit_pop[chosen]

                # searching new areas

                if best_sol <= last_sol:
                    notimproved += 1
                else:
                    last_sol = best_sol
                    notimproved = 0

                if notimproved >= 15:
                    file_aux = open(experiment_name + '/results.txt', 'a')
                    file_aux.write('\ndoomsday')
                    file_aux.close()

                    pop, fit_pop = doomsday(pop, fit_pop)
                    notimproved = 0

                best = np.argmax(fit_pop)
                std = np.std(fit_pop)
                mean = np.mean(fit_pop)

                # saves simulation state
                solutions = [pop, fit_pop]
                env.update_solutions(solutions)
                env.save_state()

            print(
                f'\nFinished {id_to_name(enemy_id)} ({env.weight_player_hitpoint}, {env.weight_enemy_hitpoint})')



fim = time.time()  # prints total execution time for experiment
print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

file = open(experiment_name + '/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log()  # checks environment state
