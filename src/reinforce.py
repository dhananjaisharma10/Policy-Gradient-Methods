import os
import sys
import gym
import time
import argparse
import os.path as osp

import keras
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from importlib import import_module
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import VarianceScaling

matplotlib.use('Agg')


class Reinforce(object):
    """Implementation of the policy gradient method REINFORCE.
    """

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here
        self.optimizer = Adam(self.cfg.LR)
        self.model.compile(optimizer=self.optimizer, loss=self.cfg.LOSS,
                           metrics=['accuracy'])
        self.mean_rewards = list()
        self.std_rewards = list()

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
        return np.array(states), np.array(actions), rewards

    def plot(self, filepath, y_label, x_label, y_values, errors):
        """Function to plot and save the mean and std of rewards.
        """
        plt.figure()
        plt.errorbar(x=list(range(len(y_values))), y=y_values, yerr=errors)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.savefig(filepath, dpi=400, bboxes_inches='tight')
        plt.close()

    def train(self, env):
        """Train the model on TRAINING_EPISODES.
        """

        def get_run_id():
            # A unique ID for a training session
            dt = datetime.now()
            run_id = dt.strftime('%m_%d_%H_%M')
            return run_id

        # Create directory for this run_id
        run_id = get_run_id()
        self.model_weights = osp.join(self.cfg.WEIGHTS, run_id)
        if not osp.exists(self.model_weights):
            os.makedirs(self.model_weights)
        start = time.time()

        print('*' * 10, 'TRAINING', '*' * 10)
        for e in range(self.cfg.TRAINING_EPISODES):
            states, actions, rewards = self.generate_episode(env)
            episode_length = len(states)
            G = [0] * episode_length
            # Find the expected return of this state,
            # beginning from the last state (excluding the terminal state)
            for state in reversed(range(episode_length)):
                # TODO: Optimize this code further.
                for ep in range(state, episode_length):
                    G[state] += self.cfg.GAMMA ** (ep - state) * rewards[ep]
            G = np.array(G).flatten()
            history = self.model.fit(states, actions,
                                     batch_size=self.cfg.BATCH_SIZE,
                                     epochs=self.cfg.EPOCHS, sample_weight=G,
                                     verbose=0)
            end = time.time()
            print('Episode {}/{} | Time elapsed: {:.2f} mins'.format(
                e + 1, self.cfg.TRAINING_EPISODES, (end - start) / 60),
                end='\r', flush=True)
            # Test after every K episodes
            if (e + 1) % self.cfg.K == 0:
                self.test(env)

    def test(self, env):
        """Test the model on K episodes.
        """

        print('\n', '*'*10, 'EVALUATION', '*'*10)
        cummulative_rewards = list()
        for e in range(self.cfg.TEST_EPISODES):
            done = False
            state = env.reset()
            episode_reward = 0
            while not done:
                actions = self.model.predict(np.array(state).reshape(1, -1))
                action = np.argmax(actions)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
            cummulative_rewards.append(episode_reward)
        # Append the mean and std of rewards
        self.mean_rewards.append(np.mean(cummulative_rewards))
        self.std_rewards.append(np.std(cummulative_rewards))
        filepath = osp.join(self.model_weights, 'rewards.jpg')
        self.plot(filepath, 'reward', '#episodes / K', self.mean_rewards,
                  self.std_rewards)


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
    pgm = Reinforce(model, cfg)
    pgm.train(env)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    tf.logging.set_verbosity(tf.logging.ERROR)
    main(sys.argv)
