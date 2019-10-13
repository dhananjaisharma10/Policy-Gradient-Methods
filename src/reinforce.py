import sys
import argparse
import gym
import keras
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from importlib import import_module
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import VarianceScaling

matplotlib.use('Agg')


class Reinforce(object):
    """Implementation of the policy gradient method REINFORCE.
    """

    def __init__(self, model, lr):
        self.model = model
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here
        self.optimizer = Adam(lr)

    def train(self, env, gamma=1.0):
        """Trains the model on a single episode using REINFORCE.
        """
        states, actions, rewards = self.generate_episode(env)
        episode_length = len(states)
        G = [0] * episode_length
        # Starting from the highest state
        for state in reversed(range(episode_length)):
            for ep in range(state, episode_length):
                G[state] += gamma ** (ep - state) * rewards[ep]
        probs = self.model(states)
        probs = probs[range(len(states)), actions]
        probs = np.log(probs)
        assert(len(probs) == len(states))
        L = np.mean(G * probs)
        self.model.compile()

    def generate_episode(self, env, render=False):
        """Generates an episode by executing the current policy
        in the given environemnt.

        Arguments:
        - env: environment

        Returns:
        - states (list)
        - actions (list)
        - rewards (list)
        """

        states = list()
        actions = list()
        rewards = list()
        done = False
        state = env.reset()
        while not done:
            prev_state = state
            acts = self.model.predict(np.array(prev_state).reshape(1, -1))
            action = np.argmax(acts)
            state, reward, done, _ = env.step(action)
            # NOTE: Terminal states will not get added.
            states.append(prev_state)
            actions.append(action)
            rewards.append(reward)
        return np.array(states), actions, rewards


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser(description='Parser for REINFORCEMENT')
    parser.add_argument('--env', dest='env', type=str, required=True,
                        help="Configuration .py file of the environment")
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)
    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    filename = args.env[:-3]
    cfg = import_module(filename)

    # Create the environment.
    env = gym.make(cfg.ENV)

    # Network architecture
    input_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    layers = list()
    for i, layer in enumerate(cfg.HIDDEN_LAYERS):
        if i == 0:
            layers.append(Dense(layer, activation=cfg.ACTIVATION,
                                kernel_initializer=VarianceScaling(
                                    distribution=cfg.KERNEL_INITIALIZER),
                                bias_initializer=cfg.BIAS_INITIALIZER,
                                input_shape=(input_dim,)))
        else:
            layers.append(Dense(layer, activation=cfg.ACTIVATION,
                                kernel_initializer=VarianceScaling(
                                    distribution=cfg.KERNEL_INITIALIZER),
                                bias_initializer=cfg.BIAS_INITIALIZER))
    layers.append(Dense(out_dim, activation='softmax',
                        kernel_initializer=VarianceScaling(
                                    distribution=cfg.KERNEL_INITIALIZER),
                        bias_initializer=cfg.BIAS_INITIALIZER))
    model = keras.Sequential(layers)
    print(model.summary())
    # TODO: Train the model using REINFORCE and plot the learning curve.
    # pgm = Reinforce(model, cfg.LR)
    # for episode in range(cfg.TRAINING_EPISODES):
    #     pgm.train(env, cfg.GAMMA)


if __name__ == '__main__':
    main(sys.argv)
