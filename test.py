import torch
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

from DQN import Agent
from Q_learning import MountainCarAgent

dqn_rewards = []

def test_DQN_trained_agent():
    env = gym.make('MountainCar-v0')
    agent = Agent(env, episodes=100, timesteps=200, batch_size=64)
    try:
        agent.load_state_dict(torch.load('dqn-nn/dqn_mountaincar-4.pth'))
        print("Modelo carregado com sucesso!")
    except FileNotFoundError:
        print("Arquivo de modelo não encontrado. Usando agente não treinado.")
    
    agent.eval()

    for ep in range(5):
        state, _ = env.reset()
        state = np.reshape(state, (1, env.observation_space.shape[0]))
        
        total_reward = 0
        steps = 0
        done = False
        
        print(f"\nIniciando episódio de teste {ep+1}")
        
        while not done:
            time.sleep(0.01)
            action = agent.get_action(state, exploration_factor=0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, (1, env.observation_space.shape[0]))
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps >= 1000:
                print("Episódio encerrado por limite de passos")
                break
        
        print(f"Episódio {ep+1} finalizado: Recompensa={total_reward}, Passos={steps}")
        dqn_rewards.append(total_reward)
    env.close()
    print("Teste concluído!")

def test_QLearning_trained_agent():
    env = gym.make('MountainCar-v0')
    
    try:
        loaded_q_table = np.load('logs/q_table_logs/MountainCar_Q_Learning-3_q_table.npy')
        print("Q-table carregada com sucesso!")
    except FileNotFoundError:
        print("Arquivo de Q-table não encontrado")
        raise ValueError()

    agent = MountainCarAgent(env, alpha=0.1, gamma=0.99, epsilon=0, epsilon_min=0, 
                            epsilon_dec=0.995, episodes=100)
    agent.Q = loaded_q_table
    q_rewards = []
    
    for ep in range(5):
        state, _ = env.reset()
        state_adj = agent.transform_state(state)
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            time.sleep(0.01)
            action = np.argmax(agent.Q[state_adj[0], state_adj[1]])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_adj = agent.transform_state(next_state)
            state_adj = next_state_adj
            total_reward += reward
            steps += 1
            
            if steps >= 1000:
                break
        
        q_rewards.append(total_reward)
    
    env.close()
    return q_rewards

def plot_mean_comparison(dqn_rewards, q_rewards):
    agents = ['DQN', 'Q-Learning']
    mean_rewards = [np.mean(dqn_rewards), np.mean(q_rewards)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(agents, mean_rewards, width=0.6, color=['blue', 'orange'])
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.set_ylabel('Média de Recompensa')
    ax.set_title('Comparação da Média de Retorno em 5 Inferências')
    ax.set_ylim(bottom=min(mean_rewards)*1.1 if min(mean_rewards) < 0 else 0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('agent_mean_comparison.png')
    plt.show()
    
    print(f"Média de recompensa DQN: {np.mean(dqn_rewards):.2f}")
    print(f"Média de recompensa Q-Learning: {np.mean(q_rewards):.2f}")

test_DQN_trained_agent()
q_rewards = test_QLearning_trained_agent()
plot_mean_comparison(dqn_rewards, q_rewards)