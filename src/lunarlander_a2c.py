"""Configuration file for Lunar Lander A2C.
"""

ENV = 'LunarLander-v2'
N = 50

HIDDEN_LAYERS = [16, 16, 16]
ACTIVATION = 'relu'
BIAS_INITIALIZER = 'zeros'
KERNEL_INITIALIZER = 'uniform'

K = 300
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.99
BATCH_SIZE = 32
EPOCHS = 1
LOSS_ACTOR = 'sparse_categorical_crossentropy'
LOSS_CRITIC = 'mse'
TRAINING_EPISODES = 50000
TEST_EPISODES = 100

# Model save directory.
WEIGHTS = './lunarlander_weights_a2c'