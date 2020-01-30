import os
import sys
import time
import argparse
import os.path as osp

import gym
import keras
import numpy as np
import tensorflow as tf

import matplotlib

from datetime import datetime
from importlib import import_module
from collections import deque
from reinforce import Reinforce

from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import VarianceScaling

matplotlib.use('Agg')


class A2C(Reinforce):
    """Implementation of N-step Advantage Actor Critic.
    This class inherits the Reinforce class
    """

    def __init__(self, action_model, critic_model, cfg):
        """Initializes A2C.
        Args:
            model: The actor model.
            lr: Learning rate for the actor model.
            critic_model: The critic model.
            cfg: Configuration file for the environment.
        """

        self.model = action_model
        self.critic_model = critic_model
        self.cfg = cfg
        self.n = cfg.N
        self.optimizer_actor = Adam(self.cfg.LR_ACTOR)
        self.model.compile(optimizer=self.optimizer_actor,
                           loss=self.cfg.LOSS_ACTOR,
                           metrics=['accuracy'])
        self.optimizer_critic = Adam(self.cfg.LR_CRITIC)
        self.critic_model.compile(optimizer=self.optimizer_critic,
                                  loss=self.cfg.LOSS_CRITIC,
                                  metrics=['accuracy'])
        self.mean_rewards = list()
        self.std_rewards = list()
        self.early_stopping = deque(maxlen=cfg.EARLY_STOPPING)

    def train(self, env):
        """Train the model on TRAINING_EPISODES.
        """

        def get_run_id():
            # A unique ID for a training session
            dt = datetime.now()
            run_id = dt.strftime('%m_%d_%H_%M') + '_{}'.format(self.n)
            return run_id

        # Create directory for this run_id
        run_id = get_run_id()
        self.model_weights = osp.join(self.cfg.WEIGHTS, run_id)
        if not osp.exists(self.model_weights):
            os.makedirs(self.model_weights)
        start = time.time()
        print('Critic N:', self.n)
        print('Actor learning rate:', self.cfg.LR_ACTOR)
        print('Critic learning rate:', self.cfg.LR_CRITIC)
        print('*' * 10, 'TRAINING', '*' * 10)
        training = True
        for e in range(self.cfg.TRAINING_EPISODES):
            states, actions, rewards = self.generate_episode(env)
            self.early_stopping.append(sum(rewards))
            # early stopping criterion
            if training and np.mean(self.early_stopping) > 200:
                training = False
                print('Early stopping criterion met at episode', e)
            # downscaling the reward
            rewards = np.array(rewards) / self.cfg.REWARD_DOWNSCALE
            episode_length = len(states)
            R = [0] * episode_length
            V = self.critic_model.predict(states)
            # Find the expected return of this state,
            # beginning from the last state (excluding the terminal state)
            temp = 0
            for state in reversed(range(episode_length)):
                R[state] = rewards[state]
                R[state] += self.cfg.GAMMA * temp
                value = 0
                if state + self.n < episode_length:
                    sub = (self.cfg.GAMMA ** self.n) * rewards[state + self.n]
                    R[state] -= sub
                    value = V[state + self.n]
                temp = R[state]
                R[state] += (self.cfg.GAMMA ** self.n) * value
            # Weight the cross entropy loss using (R - V)
            R = np.array(R).reshape(-1, 1)
            d = R - V
            d = np.squeeze(d, axis=1)
            if training:
                _ = self.model.fit(states, actions,
                                   batch_size=self.cfg.BATCH_SIZE,
                                   epochs=self.cfg.EPOCHS,
                                   sample_weight=d,
                                   verbose=0)
                _ = self.critic_model.fit(states, R,
                                          batch_size=self.cfg.BATCH_SIZE,
                                          epochs=self.cfg.EPOCHS,
                                          verbose=0)
            end = time.time()
            print('Episode {}/{} | Time elapsed: {:.2f} mins'.format(
                e + 1, self.cfg.TRAINING_EPISODES, (end - start) / 60),
                end='\r', flush=True)
            # Test after every K episodes
            if (e + 1) % self.cfg.K == 0:
                self.test(env)


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
    actor_layers = list()
    init = VarianceScaling(mode='fan_avg', distribution=cfg.KERNEL_INITIALIZER)
    # ACTOR
    for i, layer in enumerate(cfg.HIDDEN_LAYERS):
        if i == 0:
            actor_layers.append(Dense(
                layer,
                activation=cfg.ACTIVATION,
                kernel_initializer=init,
                bias_initializer=cfg.BIAS_INITIALIZER,
                input_shape=(input_dim,)))
        else:
            actor_layers.append(Dense(
                layer,
                activation=cfg.ACTIVATION,
                kernel_initializer=init,
                bias_initializer=cfg.BIAS_INITIALIZER))
    actor_layers.append(Dense(
        out_dim,
        activation='softmax',
        kernel_initializer=init))
    action_model = keras.Sequential(actor_layers)
    # CRITIC
    critic_layers = list()
    for i, layer in enumerate(cfg.HIDDEN_LAYERS):
        if i == 0:
            critic_layers.append(Dense(
                layer,
                activation=cfg.ACTIVATION,
                kernel_initializer=init,
                bias_initializer=cfg.BIAS_INITIALIZER,
                input_shape=(input_dim,)))
        else:
            critic_layers.append(Dense(
                layer,
                activation=cfg.ACTIVATION,
                kernel_initializer=init,
                bias_initializer=cfg.BIAS_INITIALIZER))
    critic_layers.append(Dense(
        1,
        kernel_initializer=init))  # final layer for critic
    critic_model = keras.Sequential(critic_layers)
    print(action_model.summary())
    print(critic_model.summary())
    pgm = A2C(action_model, critic_model, cfg)
    pgm.train(env)


if __name__ == '__main__':
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.ERROR)
    main(sys.argv)
