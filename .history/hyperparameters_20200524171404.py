import gym
from PIL import Image
import numpy as np
import sys
import gc
import os
from time import time, sleep

from random import random, randint, randrange

import pickle as p
import joblib

# Hyperparameters

NUM_EXPLORE = 20 
# 300 for Breakout

TARGET_UPDATE_PERIOD = 5000
# 1000 for Breakout

LEARNING_RATE = 0.00025

DISCOUNT_FACTOR = 0.99

BATCH_SIZE = 32

NOOPMAX = 10

MAX_EXPERIENCES = 40000 # Memory Size

MIN_EPSILON = 0.1
# 0.05 for Breakout

DECAY_RATE = 0.9999

TOTAL_LIVES = 5 # Depend on the game
# 5 for Breakout

K = 4 # Change this!
# Not used in Breakout

# Adam Optimizer
ADAM_Opt = Adam(lr=LEARNING_RATE)
