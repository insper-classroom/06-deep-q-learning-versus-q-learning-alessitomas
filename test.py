import torch
import gymnasium as gym
import numpy as np
import time


from main import Agent

def test_trained_agent():
    env = gym.make('MountainCar-v0', render_mode='human')
    
    
    agent = Agent(env, episodes=100, timesteps=200, batch_size=64)
    
    
    try:
        agent.load_state_dict(torch.load('dqn_mountaincar.pth'))
        print("Modelo carregado com sucesso!")
    except FileNotFoundError:
        print("Arquivo de modelo não encontrado. Usando agente não treinado.")
    
    # Configurar para modo de avaliação (sem exploração)
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
    
    env.close()
    print("Teste concluído!")


test_trained_agent()