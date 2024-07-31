import random
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layer import Dense,Flatten

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env =gym.make("CartPole-v1",render_mode="human")

states = env.observation_space.shape[0] #possible states in current env
actions = env.action_space.n #number of various actions that can be performed in the current env



episodes = 10
for episode in range(1,episodes+1):
    state=env.reset()
    done=False
    score=0

    while not done:
        action=random.choice([0,1]) #0 for left , 1 for right
        _,reward,done,_=env.step(action) #step returns 4 values
        score+=reward
        env.render()

    print(f"Episode {episode}, Score: {score}")

env.close()
        