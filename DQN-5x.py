from DQN import Agent
import torch
import gymnasium as gym
import numpy as np

for i in range(5):

    env = gym.make('MountainCar-v0', render_mode=None)  
    agent = Agent(env, episodes=5000, timesteps=1000, batch_size=64)
    rewards = agent.train_agent()
    np.save(f'dqn-data/learning_curve_data-{i}.npy', rewards)
    torch.save(agent.state_dict(), f'dqn-nn/dqn_mountaincar-{i}.pth')