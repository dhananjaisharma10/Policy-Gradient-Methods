"""Configuration file for Lunar Lander REINFORCEMENT.
"""

ENV = 'LunarLander-v2'

HIDDEN_LAYERS = [16, 16, 16]
ACTIVATION = 'relu'
BIAS_INITIALIZER = 'zeros'
KERNEL_INITIALIZER = 'uniform'

LR = 5e-4
GAMMA = 1.0
TRAINING_EPISODES = 50000
