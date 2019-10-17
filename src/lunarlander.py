"""Configuration file for Lunar Lander REINFORCEMENT.
"""

ENV = 'LunarLander-v2'

HIDDEN_LAYERS = [16, 16, 16]
ACTIVATION = 'relu'
BIAS_INITIALIZER = 'zeros'
KERNEL_INITIALIZER = 'uniform'

K = 300
LR = 5e-4
GAMMA = 0.99
BATCH_SIZE = 32
EPOCHS = 1
LOSS = 'sparse_categorical_crossentropy'
TRAINING_EPISODES = 50000
TEST_EPISODES = 100
EARLY_STOPPING = 50

# Model save directory.
WEIGHTS = './lunarlander_weights'
