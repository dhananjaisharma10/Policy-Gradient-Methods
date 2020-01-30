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
from collections import deque

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
        self.optimizer = Adam(self.cfg.LR)
        self.model.compile(optimizer=self.optimizer, loss=self.cfg.LOSS,
                           metrics=['accuracy'])
        self.mean_rewards = list()
        self.std_rewards = list()
        # stores a definite number of past rewards to check for early stopping
        self.early_stopping = deque(maxlen=cfg.EARLY_STOPPING)

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
            probs = self.model.predict(np.array(prev_state).reshape(1, -1))
            # Stochastic policy for choosing actions
            action = np.random.choice(list(range(env.action_space.n)),
                                      p=probs[0])
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
        """Train the model.
        """

        def get_run_id():
            # A unique ID for a training session
            dt = datetime.now()
            run_id = dt.strftime('%m_%d_%H_%M') + '_r'
            return run_id

        # Create directory for this run_id
        run_id = get_run_id()
        self.model_weights = osp.join(self.cfg.WEIGHTS, run_id)
        if not osp.exists(self.model_weights):
            os.makedirs(self.model_weights)
        start = time.time()

        print('*' * 10, 'TRAINING', '*' * 10)
        training = True
        early_stop = 0
        for e in range(self.cfg.TRAINING_EPISODES):
            states, actions, rewards = self.generate_episode(env)
            # Check for early stopping criterion
            self.early_stopping.append(sum(rewards))
            if training and np.mean(self.early_stopping) > 200:
                training = False
                early_stop = e
            episode_length = len(states)
            G = [0] * episode_length
            # Find the expected return of this state,
            # beginning from the last state (excluding the terminal state)
            for state in reversed(range(episode_length)):
                G[state] = rewards[state]
                if state < episode_length - 1:
                    G[state] += self.cfg.GAMMA * G[state + 1]
            # Normalize the expected returns
            G = np.array(G)
            G = (G - np.mean(G)) / np.std(G)
            if training:
                _ = self.model.fit(states, actions,
                                   batch_size=self.cfg.BATCH_SIZE,
                                   epochs=self.cfg.EPOCHS,
                                   sample_weight=G,
                                   verbose=0)
            end = time.time()
            print('Episode {}/{} | Time elapsed: {:.2f} mins'.format(
                e + 1, self.cfg.TRAINING_EPISODES, (end - start) / 60),
                end='\r', flush=True)
            # Test after every K episodes
            if (e + 1) % self.cfg.K == 0:
                print('\nEarly stopping episode:', early_stop)
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
        mean_reward = np.mean(cummulative_rewards)
        std_reward = np.std(cummulative_rewards)
        self.mean_rewards.append(mean_reward)
        self.std_rewards.append(std_reward)
        meanpath = osp.join(self.model_weights, 'mean_rewards.npy')
        stdpath = osp.join(self.model_weights, 'std_rewards.npy')
        np.save(meanpath, self.mean_rewards)
        np.save(stdpath, self.std_rewards)
        print('Mean reward: {:.3f} | Std: {:.3f}'.format(mean_reward,
              std_reward))
        filepath = osp.join(self.model_weights, 'rewards.png')
        self.plot(filepath, 'reward', '#episodes / K', self.mean_rewards,
                  self.std_rewards)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser(description='Parser for REINFORCEMENT')
    parser.add_argument('--env', dest='env', type=str, required=True,
                        help="Configuration .py file of the environment")
    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    filename = args.env[:-3]
    cfg = import_module(filename)

    # Create the environment.
    env = gym.make(cfg.ENV)

    # Setting the session to allow growth, so it doesn't allocate all GPU
    # memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    # Network architecture
    input_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    layers = list()
    init = VarianceScaling(mode='fan_avg', distribution=cfg.KERNEL_INITIALIZER)
    for i, layer in enumerate(cfg.HIDDEN_LAYERS):
        if i == 0:
            layers.append(Dense(layer, activation=cfg.ACTIVATION,
                                kernel_initializer=init,
                                bias_initializer=cfg.BIAS_INITIALIZER,
                                input_shape=(input_dim,)))
        else:
            layers.append(Dense(layer, activation=cfg.ACTIVATION,
                                kernel_initializer=init,
                                bias_initializer=cfg.BIAS_INITIALIZER))
    layers.append(Dense(out_dim, activation='softmax',
                        kernel_initializer=init,
                        bias_initializer=cfg.BIAS_INITIALIZER))
    model = keras.Sequential(layers)
    print(model.summary())
    pgm = Reinforce(model, cfg)
    pgm.train(env)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable AVX/FMA warnings
    tf.logging.set_verbosity(tf.logging.ERROR)
    main(sys.argv)
