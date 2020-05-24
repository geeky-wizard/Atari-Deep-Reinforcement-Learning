import gym
from PIL import Image
import numpy as np
import sys
import gc
import os
from time import time, sleep

import pickle as p
import joblib

from hyperparameters import *
from Agent import *
from CNN import *
from helper import *

def test(game):
    pass

def initial_exploration(game):
    log = open("log_"+game+".txt", 'w')
    log.write("\n\n=================================  Starting Exploration  ==============================\n")
    
#     current_lives = TOTAL_LIVES  # Total Number of lives
    avg_score = 0
    
    for episode in range(NUM_EXPLORE):
        timer = time()
        done = False
        
        env.reset()
        obsv,_,_,_ = env.step(0)
        obsv = process_state(obsv)
        current_state = np.array([obsv, obsv, obsv, obsv])
        
        for _ in range(randint(1,NOOPMAX)):
            obsv,_,_,_ = env.step(0)
            obsv = process_state(obsv)
            next_state = get_next_state(current_state, obsv)
            current_state = next_state
        
        steps = 0
        score = 0
        
        while not done:
            steps+=1

            action = randint(1, K)

            obsv, reward, done, info = env.step(action)
            obsv = process_state(obsv)

            next_state = get_next_state(current_state, obsv)

            clipped_reward = np.clip(reward, -1, 1)
            # clipped_reward = reward
            agent.experience_gain(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)
            
            
            current_state = next_state

            score += clipped_reward
        
        timer = time() - timer
        avg_score = (avg_score + score)/2 if episode != 0 else score
        log.write(str(episode+1) + "\tTotalReward = " + str(score) + "\tSteps: " + str(steps) + "\tMoving Avg: {:.2f}".format(avg_score) + "\tTime: %d" % int(timer/60) + ":{:.0f} \n".format((timer % 60)))
        print(episode+1, "\tTotalReward = ", score, "\tSteps: ", steps, "\tMoving Avg: {:.2f}".format(avg_score),"\tTime: %d" % (timer/60), "\b:{:.0f}".format((timer % 60)))
    
    log.close()
    agent.save_state()
    print("\nEXPLORATION STEP COMPLETED\n")
    pass

def train(game):
    pass

if __name__ == "__main__":
    print('Train or Test any atari games on this DDQN Network\n')
    # For games like breakout, where manual fire is necessary only a part of code in Training and Exploration needs to be changed.
    try:
        game_name = sys.argv[1]
    except IndexError:
        game_name = "Breakout"

    game = game_name + "-v4"
    while(1):
        choice = input("1. Train Agent\n2. Run Test\n3. View results of pre-trained weights\n: ")
        if  choice == '1':
            if os.path.exists("Experiences"+game):
                train(game)
            else:
                initial_exploration(game)
                train(game)
            break
        elif choice == '2':
            test(game)
            break
        elif choice == '3':

            # Pass graph generation function open using matplotlib
            break
        else :
            print('Please Enter a valid choice\n')