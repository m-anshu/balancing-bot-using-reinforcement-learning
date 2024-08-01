import random
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import optimizers
import numpy as np 

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# env =gym.make("CartPole-v1",render_mode="human")
env =gym.make("CartPole-v1") # render mode can be set to human if visualize =True to visualize training process
states = env.observation_space.shape[0] #possible states in current env
actions = env.action_space.n #number of various actions that can be performed in the current env

model=Sequential()
model.add(Flatten(input_shape=(1,states)))
model.add(Dense(24,activation="relu"))
model.add(Dense(24,activation="relu"))
model.add(Dense(actions,activation="linear"))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000,window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)
optimizer = optimizers.legacy.Adam(learning_rate=0.001) #we use a legacy version of Adam as it seems to not work otherwise :(
agent.compile(optimizer, metrics=["mae"]) # mean absolute error
agent.fit(env,nb_steps=10000,visualize=False,verbose=1) #visualize can be set to true if you want to see the training process,additionally set render_mode="human" in the environment

results = agent.test(env, nb_episodes=10,visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()


        