import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime
import random



def generate_cumulative_reward_log(cumulative_reward_history, timesteps_per_episode, datetime_str, run_name):
    
    os.makedirs("logs/cum_reward_logs", exist_ok=True)
    
    if run_name:
        filename = f"logs/cum_reward_logs/{run_name}_cumulative_reward_log.txt"
    else:  
        filename = f"logs/cum_reward_logs/{datetime_str}_cumulative_reward_log.txt"
    
    with open(filename, "w") as f:
        for episode in range(len(cumulative_reward_history)):
            f.write(f"Episode: {episode + 1} - Reward: {cumulative_reward_history[episode]} - Moves: {timesteps_per_episode[episode]} \n")
    return filename

def generate_q_table(q_table, datetime_str, run_name):
    
    os.makedirs("logs/q_table_logs", exist_ok=True)
    
    if run_name:
        filename = f"logs/q_table_logs/{run_name}_q_table.npy"
    else:
        filename = f"logs/q_table_logs/{datetime_str}_q_table.npy"
    
    
    np.save(filename, q_table)
    return filename



def read_rewards_from_file(filename):
    episodes = []
    rewards = []
    moves = []
    with open(filename, 'r') as file:
        for line in file:
            match = re.search(r"Episode: (\d+) - Reward: (-?\d+\.?\d*) - Moves: (\d+)", line)
            if match:
                episodes.append(int(match.group(1)))
                rewards.append(float(match.group(2)))
                moves.append(int(match.group(3)))
    return episodes, rewards, moves

def plot_learning_curve(input_filename, datetime_str, run_name):
    
    os.makedirs("results", exist_ok=True)
    
    episodes, rewards, moves = read_rewards_from_file(input_filename)
    
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(episodes, rewards, marker='.', linestyle='-', color='b')
    plt.title(f'Curva de Aprendizado do Agente: {run_name}', fontsize=14)
    plt.xlabel('Número do Episódio', fontsize=12)
    plt.ylabel('Recompensa Acumulada', fontsize=12)
    plt.grid(True)
    
    
    plt.subplot(2, 1, 2)
    plt.plot(episodes, moves, marker='.', linestyle='-', color='r')
    plt.title(f'Passos por Episódio: {run_name}', fontsize=14)
    plt.xlabel('Número do Episódio', fontsize=12)
    plt.ylabel('Número de Passos', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    
    if run_name:
        filename = f"{run_name}_learning_curve.png"
    else:
        filename = f"{datetime_str}_learning_curve.png"
    
    output_path = os.path.join("results", filename)
    plt.savefig(output_path)
    plt.close()
    return output_path



class MountainCarAgent:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        
        
        self.num_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
        self.num_states = np.round(self.num_states, 0).astype(int) + 1
        
        
        self.Q = np.zeros([self.num_states[0], self.num_states[1], env.action_space.n])
    
    def transform_state(self, state):
        """Convert continuous state to discrete state indices"""
        state_adj = (state - self.env.observation_space.low) * np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        
        state_adj[0] = max(0, min(state_adj[0], self.num_states[0] - 1))
        state_adj[1] = max(0, min(state_adj[1], self.num_states[1] - 1))
        return state_adj
    
    def select_action(self, state_adj):
        """Epsilon-greedy action selection"""
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[state_adj[0], state_adj[1]])
        return self.env.action_space.sample()
    
    def update_q_value(self, state_adj, action, reward, next_state_adj):
        """Update Q-value using Q-learning update rule"""
        best_next_action = np.argmax(self.Q[next_state_adj[0], next_state_adj[1]])
        td_target = reward + self.gamma * self.Q[next_state_adj[0], next_state_adj[1], best_next_action]
        td_error = td_target - self.Q[state_adj[0], state_adj[1], action]
        self.Q[state_adj[0], state_adj[1], action] += self.alpha * td_error
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec
    
    def train(self, max_steps_per_episode=1000, run_name=None):
        """Train the agent using Q-learning"""
        cumulative_reward_history = []
        timesteps_per_episode = []
        epsilon_history = []
        
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            state_adj = self.transform_state(state)
            cumulative_reward = 0
            
            for t in range(max_steps_per_episode):
                
                action = self.select_action(state_adj)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state_adj = self.transform_state(next_state)
                
                
                self.update_q_value(state_adj, action, reward, next_state_adj)
                
                
                state_adj = next_state_adj
                cumulative_reward += reward
                
                if terminated or truncated:
                    break
            
            
            self.decay_epsilon()
            
            
            timesteps_per_episode.append(t+1)
            cumulative_reward_history.append(cumulative_reward)
            epsilon_history.append(self.epsilon)
            
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(cumulative_reward_history[-100:])
                avg_steps = np.mean(timesteps_per_episode[-100:])
                print(f"Episode: {episode + 1}/{self.episodes}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Epsilon: {self.epsilon:.4f}")
        
        
        datetime_str = datetime.strftime(datetime.now(), '%d_%m_%Y_%H_%M_%S')
        cum_filepath = generate_cumulative_reward_log(cumulative_reward_history, timesteps_per_episode, datetime_str, run_name)
        q_table_path = generate_q_table(self.Q, datetime_str, run_name)
        plot_path = plot_learning_curve(cum_filepath, datetime_str, run_name)
        
        print(f"Training completed. Files saved: {cum_filepath}, {q_table_path}, {plot_path}")
        return self.Q
    
    def test(self, num_episodes=10, render=True):
        """Test the trained agent"""
        total_rewards = 0
        total_steps = 0
        wins = 0
        wins_mean_steps = None
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state_adj = self.transform_state(state)
            episode_reward = 0
            steps = 0
        
            done = False
            while not done:
                
                action = np.argmax(self.Q[state_adj[0], state_adj[1]])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state_adj = self.transform_state(next_state)
                
                if render:
                    self.env.render()
                
                state_adj = next_state_adj
                episode_reward += reward
                steps += 1
                
                if terminated or truncated:
                    if terminated:
                        if wins_mean_steps is None:
                            wins_mean_steps = steps
                        else: 
                            wins_mean_steps += steps
                        wins += 1
                    done = True
            
            total_rewards += episode_reward
            total_steps += steps
            print(f"Test Episode {episode+1}: Reward = {episode_reward}, Steps = {steps}")
        
        avg_reward = total_rewards / num_episodes
        avg_steps = total_steps / num_episodes
        print(f"Testing completed. Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}")
        print(f"Wins {wins}")
        if wins_mean_steps is not None:
            wins_mean_steps /= wins
        return avg_reward, avg_steps, wins, wins_mean_steps 



if __name__ == "__main__":
    
    alpha = 0.9        
    gamma = 0.99       
    epsilon = 1.0      
    epsilon_min = 0.01 
    epsilon_dec = 0.999
    episodes = 10000
    
    
    env = gym.make('MountainCar-v0')
    
    
    print("Environment Information:")
    print('State space:', env.observation_space)
    print('Action space:', env.action_space)
    print('State low bound:', env.observation_space.low)
    print('State high bound:', env.observation_space.high)
    
    
    agent = MountainCarAgent(
        env=env,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_dec=epsilon_dec,
        episodes=episodes
    )
    
    print(f"Training agent with hyperparameters:")
    print(f"  - Learning rate (alpha): {alpha}")
    print(f"  - Discount factor (gamma): {gamma}")
    print(f"  - Initial exploration rate (epsilon): {epsilon}")
    print(f"  - Minimum exploration rate (epsilon_min): {epsilon_min}")
    print(f"  - Exploration decay rate (epsilon_dec): {epsilon_dec}")
    print(f"  - Number of episodes: {episodes}")
    
    
    q_table = agent.train(max_steps_per_episode=200, run_name="MountainCar_Q_Learning")
    
    
    try:
        test_env = gym.make('MountainCar-v0', render_mode='human')
        agent.env = test_env
        agent.test(num_episodes=5)
    except Exception as e:
        print(f"Could not render test environment: {e}")
        print("Running without visualization")
        agent.test(num_episodes=5, render=False)
    
    
    env.close()
    try:
        test_env.close()
    except:
        pass