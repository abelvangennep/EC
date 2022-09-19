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

sys.path.insert(0, 'evoman')
from environment import Environment
from neat_controller import player_controller
# from demo_controller import player_controller
from Neat import Node_Gene, Connection_Gene, initialize_network, Individual
from neat_selection import parent_selection
from Neat_speciation import speciation
from NEAT_crossover import crossover
# imports other libs
import numpy as np


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
for en in range(1, 2):
    #start with population, create 10 random individuals (1 for training now)
    pop = [Individual(initialize_network()) for i in range(10)]
    for i in range(5): #number of generations
        for pcont in pop:
            # Update the enemy
            env.update_parameter('enemies', [en])
            sol = pcont
            # Load specialist controller
            # sol = np.loadtxt('solutions_demo/demo_' + str(en) + '.txt')
            # print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(en) + ' \n')
            vfitness, vplayerlife, venemylife, vtime = env.play(sol)
            pcont.set_fitness(vfitness)
            print('My fitnes: ', vfitness)
        #Selection, crossover, mutation, evaluation loop for 10 generations
        #----- Elena -----
        species = speciation(pop) #The speciation function takes whole population as list of individuals and returns
        # a list of lists with individuals [[1,2], [4,5,8], [3,6,9,10], [7]] for example with 10 individuals
        #----- Tim -------
        parents = parent_selection(species) #This function returns pairs of parents which will be mated. In total
        # the number of pairs equal to the number of offsprings we want to generate
        #----- Abel -------
        children = []
        for pair in parents:
            children.append(crossover(pair[0], pair[1])) #for loop needed to cross each pair of parents
        #-----      -------
        #mut_children = mutate(children)
        #----- Tim  ------
        #evaluate/run for whole new generation and assign fitness value
        pop = children
        #repeat loop

    #Evaluate

