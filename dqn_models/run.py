# Use to demo models

import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
import tensorflow as tf
import os

random.seed(0)
tf.random.set_seed(0)
np.random.seed(0)


class DQN:
    def __init__(self,
                 action_num, state_shape,
                 learning_rate=0.001, reward_decay=0.9,
                 e_greedy_min=0.01, e_greedy_increment=(0.5-0.01)/100000,
                 memory_size=4096, batch_size=32, replace_iterations=500):
        self.action_num = action_num
        self.state_shape = state_shape
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy = 0.5 if e_greedy_increment is not None else e_greedy_min
        self.e_greedy_min = e_greedy_min
        self.e_greedy_increment = e_greedy_increment
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_iterations = replace_iterations
        self.pos_memory = deque(maxlen=int(0.5*memory_size))
        self.neg_memory = deque(maxlen=memory_size-int(0.5*memory_size))
        self.learning_step = 0
        self.eval_nn = self.__build_nn()
        self.target_nn = self.__build_nn()

    def __build_nn(self):
        model = Sequential([
            Dense(64, activation="relu", input_shape=self.state_shape),
            Dense(64, activation="relu"),
            Dense(self.action_num, activation="relu"),
        ])
        model.compile(loss="mean_squared_error",
                      optimizer="adam", metrics=["accuracy"])
        return model

    def get_action(self, state, mode):
        if mode == "best":
            actions = self.eval_nn.predict(state)
            return actions.argmax()
        if mode == "e_greedy":
            if self.e_greedy > self.e_greedy_min:
                self.e_greedy -= self.e_greedy_increment
            else:
                self.e_greedy = self.e_greedy_min
            if np.random.uniform() > self.e_greedy:
                actions = self.eval_nn.predict(state)
                return actions.argmax()
            else:
                return np.random.randint(0, self.action_num)
        print("ERROR: mode doesn't exist.")
        return 0

    def load_weights(self, name):
        self.eval_nn.load_weights(name, by_name=True)


path = input("Demo dqn, nature_dqn or ddqn?\n")
path += "/record/"
files = os.listdir(path)
files.sort()

print("\nInitialing environment and model\n")
game_name = "CartPole-v0"
env = gym.make(game_name)
env.seed(0)
model = DQN(env.action_space.n, env.observation_space.shape)

print("\nDemo\n")
for file in files:
    if file[0] == '.':
        continue
    model.load_weights(path + file)
    for j in range(1):
        total_reward = 0
        state = env.reset()
        while True:
            env.render()
            action = model.get_action(state[None, :], "best")
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            if done:
                break
            state = next_state
        print(file, ", reward: ", total_reward, sep='')
env.close()
