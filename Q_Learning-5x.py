from Q_learning import MountainCarAgent
import gymnasium as gym

alpha = 0.1        
gamma = 0.99       
epsilon = 1.0      
epsilon_min = 0.01 
epsilon_dec = 0.995 
episodes = 10000   

    
env = gym.make('MountainCar-v0')

agent = MountainCarAgent(
    env=env,
    alpha=alpha,
    gamma=gamma,
    epsilon=epsilon,
    epsilon_min=epsilon_min,
    epsilon_dec=epsilon_dec,
    episodes=episodes
)
for i in range(5):
    q_table = agent.train(max_steps_per_episode=200, run_name=f"MountainCar_Q_Learning-{i}")
