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
number_generations = 3
population_size = 10
mutation_prob =0
compat_threshold =0
link_insert_prob =0
node_insert_prob =0


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


run_neat(number_generations = 15, population_size = 60, compat_threshold = 5)