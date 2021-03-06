import gym
from PIL import Image
import numpy as np
from Agent import Agent
import sys
import gc
from time import time, sleep

def test():
    pass

def train():
    pass

def main():
    pass

if __name__ == "__main__":
    print('Train or Test any atari games on this DDQN Network\n')
    try:
        game_name = sys.argv[1]
    except IndexError:
        game_name = "Breakout"
    while(1):
        choice = input("1. Train Agent\n2. Run Test\n3. View results of pre-trained weights\n: ")
        if  choice == '1':
            train(game_name)
            break
        elif choice == '2':
            test(game_name)
            break
        elif choice == '3':
            # Pass graph generation function open using matplotlib
            break
        else :
            print('Please Enter a valid choice\n')