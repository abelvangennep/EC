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
from demo_controller import player_controller
from Neat import Node_Gene, Connection_Gene, initialize_network, Individual
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
for en in range(1, 2):
    # Update the enemy
    env.update_parameter('enemies', [en])
    #net = initialize_network()
    #sol = Individual(net)
    # Load specialist controller
    sol = np.loadtxt('solutions_demo/demo_' + str(en) + '.txt')
    print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(en) + ' \n')
    vfitness, vplayerlife, venemylife, vtime = env.play(sol)
print('My fitnes: ', vfitness)
