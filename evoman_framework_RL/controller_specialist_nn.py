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

from hyperopt import fmin, tpe, hp, Trials

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
#Parameters
# number_generations = 3
# population_size = 10
# mutation_prob =0
# compat_threshold =0
# link_insert_prob =0
# node_insert_prob =0


def run_neat(number_generations = 3, population_size = 10,compat_threshold = 2,
            weight_mutation_lambda = 3, link_insertion_prob=.05, node_insertion_prob=.05, enemy=[1]):
    for en in enemy:
        results = np.zeros(((number_generations+1)*population_size, 9)) #Generation, Individual, Parents, Species, Fitness, time, avg.gen fitness, avg.gen dist
        overview = np.zeros((number_generations,2))
        # Update the enemy
        env.update_parameter('enemies', [en])
        #start with population, create 10 random individuals (1 for training now)
        pop = [Individual(initialize_network(), i) for i in range(population_size)]
        highest_innov_id = 101
        id_node = 26
  
        for gen in range(number_generations): #number of generations
            start_gen = time.time()
            #print('---- Starting with generation ', gen)
            fitnesses = []
            for pcont in pop:
                #print('Evaluating individual ', pcont.get_id())
                start_ind = time.time()
                vfitness, vplayerlife, venemylife, vtime = env.play(pcont)
                pcont.set_fitness(calc_fitness_value(vplayerlife, venemylife, vtime)+100) # no negative fitness values
                fitnesses.append(calc_fitness_value(vplayerlife, venemylife, vtime))
                #print('Fitness value: ', calc_fitness_value(vplayerlife, venemylife, vtime), ' time elapsed: ', time.time()-start_ind)
                results[gen * population_size+pcont.get_id(),0] = gen
                results[gen * population_size + pcont.get_id(), 1] = pcont.get_id()
                results[gen * population_size + pcont.get_id(), 5] = vfitness
                results[gen * population_size + pcont.get_id(), 6] = time.time()-start_ind

            overview[gen,0] = sum(fitnesses)/len(fitnesses)
            overview[gen,1] = calc_avg_dist(pop)
            results[gen*population_size,7] = sum(fitnesses)/len(fitnesses)
            results[gen * population_size, 8] = calc_avg_dist(pop)

            species = speciation(pop,compat_threshold) #The speciation function takes whole population as list of individuals and returns # a list of lists with individuals [[1,2], [4,5,8], [3,6,9,10], [7]] for example with 10 individuals
            #add species information to individual
            for m in range(len(species)):
                for j in range(len(species[m])):
                    results[gen*population_size+species[m][j].get_id(),3] = m
            print('There were ', len(species), ' different species identified in generation ', gen)
            parents = parent_selection(species) #This function returns pairs of parents which will be mated. In total the number of pairs equal to the number of offsprings we want to generate
            children = []
 
            for temp, pair in enumerate(parents):
                children.append(crossover(pair[0], pair[1])) #for loop needed to cross each pair of parents
                #write parents in result table
                results[(gen+1)*population_size+temp,2] = gen*100+min(pair[0].get_id(),pair[1].get_id())*10+max(pair[0].get_id(),pair[1].get_id())
                temp += 1
            for m in range(len(children)):
                children[m], id_node, highest_innov_id, string = mutate(children[m], id_node, highest_innov_id, weight_mutation_lambda, link_insertion_prob, node_insertion_prob)
                children[m].set_id(m)
                results[(gen + 1) * population_size + m, 4] = string

            #evaluate/run for whole new generation and assign fitness value
            pop = children
            print('Generation ', gen, ' took ', time.time()-start_gen, ' seconds to elapse. Highest fitness value was ', max(fitnesses) )


        results_df = pd.DataFrame(results, columns = ['Generation', 'Individual', 'Parents', 'Species', 'Mutation', 'Fitness', 'Time elapsed', 'Avg. fitness', 'Avg. distance'])
        results_df.to_csv('test2.csv')
        print(results_df)


# run_neat(number_generations = 15, population_size = 60, compat_threshold = 5)


def neat_optimizer(number_generations, population_size, weight_mutation_lambda, compat_threshold,link_insert_prob,node_insert_prob, enemy):
    for en in enemy:
        overview = np.zeros((number_generations,2))
        # Update the enemy
        env.update_parameter('enemies', [en])
        #start with population, create 10 random individuals (1 for training now)
        pop = [Individual(initialize_network(), i) for i in range(population_size)]
        highest_innov_id = 101
        id_node = 26
        for gen in range(number_generations): #number of generations
            
            fitnesses = []
            for pcont in pop:
                vfitness, vplayerlife, venemylife, vtime = env.play(pcont)
                pcont.set_fitness(calc_fitness_value(vplayerlife, venemylife, vtime)+100) # no negative fitness values
                fitnesses.append(calc_fitness_value(vplayerlife, venemylife, vtime))

            overview[gen,0] = sum(fitnesses)/len(fitnesses)
            overview[gen,1] = calc_avg_dist(pop)

            species = speciation(pop,compat_threshold) #The speciation function takes whole population as list of individuals and returns # a list of lists with individuals [[1,2], [4,5,8], [3,6,9,10], [7]] for example with 10 individuals
           
            #add species information to individual
            parents = parent_selection(species) #This function returns pairs of parents which will be mated. In total the number of pairs equal to the number of offsprings we want to generate
            children = []
 
            for temp, pair in enumerate(parents):
                children.append(crossover(pair[0], pair[1])) #for loop needed to cross each pair of parents
                #write parents in result table
                temp += 1
            for m in range(len(children)):
                children[m], id_node, highest_innov_id, string = mutate(children[m], id_node, highest_innov_id,weight_mutation_lambda, link_insert_prob, node_insert_prob)
                children[m].set_id(m)

            #evaluate/run for whole new generation and assign fitness value
            pop = children
            max_value = max(fitnesses)
            print('fitness:', max_value, "venemylife", venemylife)

    return max_value

def neat_iterations(parameters):
    num_iterations = 3
    number_generations = 10
    population_size = int(parameters['population_size'])
    weight_mutation_lambda = parameters['weight_mutation_lambda']
    compat_threshold = parameters['compat_threshold']
    link_insert_prob = parameters['link_insert_prob']
    node_insert_prob = parameters['node_insert_prob']
    enemy=[1]

    print(parameters)
    
    best_fitnesses = []

    for iteration in range(num_iterations):
        print(iteration)
        best_fitnesses.append(neat_optimizer(number_generations, population_size, weight_mutation_lambda,compat_threshold,
            link_insert_prob,node_insert_prob, enemy))

    return { 'loss':-np.mean(best_fitnesses),
        # -- store other results like this
        'eval_time': time.time(),
        'loss_variance': np.var(best_fitnesses)}


space = hp.choice('Type_of_model',[{
        'population_size': hp.quniform("population_size_3", 50, 100, 1),
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