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
from Neat import Node_Gene, Connection_Gene, initialize_network, Individual
from neat_sel import parent_selection
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
                  speed="normal",
                  enemymode="static",
                  level=2)



# tests saved demo solutions for each enemy
#for i in range(10):
number_generations = 10
population_size = 10
for en in range(1, 2):
    results = np.zeros(((number_generations+1)*population_size, 9)) #Generation, Individual, Parents, Species, Fitness, time, avg.gen fitness, avg.gen dist
    overview = np.zeros((number_generations,2))
    # Update the enemy
    env.update_parameter('enemies', [en])
    #start with population, create 10 random individuals (1 for training now)
    pop = [Individual(initialize_network(), i) for i in range(population_size)]
    highest_innov_id = 101
    id_node = 26
    gen = 0
    for i in range(number_generations): #number of generations
        start_gen = time.time()
        print('---- Starting with generation ', gen)
        fitnesses = []
        for pcont in pop:
            print('Evaluating individual ', pcont.get_id())
            start_ind = time.time()
            vfitness, vplayerlife, venemylife, vtime = env.play(pcont)
            pcont.set_fitness(vfitness+100) # no negative fitness values
            fitnesses.append(vfitness)
            print('Fitness value: ', vfitness, ' time elapsed: ', time.time()-start_ind)
            results[gen*population_size+pcont.get_id(),0] = gen
            results[gen * population_size + pcont.get_id(), 1] = pcont.get_id()
            results[gen * population_size + pcont.get_id(), 5] = vfitness
            results[gen * population_size + pcont.get_id(), 6] = time.time()-start_ind

        overview[gen,0] = sum(fitnesses)/len(fitnesses)
        overview[gen,1] = calc_avg_dist(pop)
        results[gen*population_size,7] = sum(fitnesses)/len(fitnesses)
        results[gen * population_size, 8] = calc_avg_dist(pop)

        species = speciation(pop) #The speciation function takes whole population as list of individuals and returns # a list of lists with individuals [[1,2], [4,5,8], [3,6,9,10], [7]] for example with 10 individuals
        #add species information to individual
        for m in range(len(species)):
            for j in range(len(species[m])):
                results[gen*population_size+species[m][j].get_id(),3] = m
        print('There were ', len(species), ' different species identified in generation ', gen)
        parents = parent_selection(species) #This function returns pairs of parents which will be mated. In total the number of pairs equal to the number of offsprings we want to generate
        children = []
        temp = 0
        for pair in parents:
            children.append(crossover(pair[0], pair[1])) #for loop needed to cross each pair of parents
            #write parents in result table
            results[(gen+1)*population_size+temp,2] = gen*100+min(pair[0].get_id(),pair[1].get_id())*10+max(pair[0].get_id(),pair[1].get_id())
            temp += 1
        for m in range(len(children)):
            id_node, highest_innov_id, string = mutate(children[m], id_node, highest_innov_id)
            children[m].set_id(m)
            results[(gen + 1) * population_size + m, 4] = string

        #evaluate/run for whole new generation and assign fitness value
        pop = children
        print('Generation ', gen, ' took ', time.time()-start_gen, ' seconds to elapse. Highest fitness value was ', max(fitnesses) )
        gen+=1
        #repeat loop
    results_df = pd.DataFrame(results, columns = ['Generation', 'Individual', 'Parents', 'Species', 'Mutation', 'Fitness', 'Time elapsed', 'Avg. fitness', 'Avg. distance'])
    results_df.to_csv('test1.csv')
    print(results_df)
    #Evaluate

