import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import numpy as np
from collections import deque

class Agent(nn.Module):
    def __init__(self, env: gym.Env, episodes, timesteps, batch_size):
        super(Agent, self).__init__()
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.model = nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )

        # Otimizador e função de perda
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Hiperparâmetros
        self.episodes = episodes
        self.timesteps = timesteps
        self.memory = deque(maxlen=10000)
        self.batch_size = batch_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_dec = 0.995

        
        # Configurar o dispositivo (CPU ou GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        return self.model(x)
        
    def get_action(self, state, exploration_factor=None):
        if exploration_factor is None:
            exploration_factor = self.epsilon
            
        rv = random.uniform(0, 1)
        if rv < exploration_factor:
            return self.env.action_space.sample()
            
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Predict usando PyTorch
        self.eval()  # Modo de avaliação
        with torch.no_grad():
            q_values = self(state_tensor)
        
        # Retornar a ação com maior valor Q
        return torch.argmax(q_values).item()

    def update(self, states, targets):
        # Converter para tensores
        states_tensor = torch.FloatTensor(states).to(self.device)
        targets_tensor = torch.FloatTensor(targets).to(self.device)
        
        # Treinar a rede
        self.train()  # Modo de treinamento
        self.optimizer.zero_grad()
        predictions = self(states_tensor)
        loss = self.criterion(predictions, targets_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def add_exp(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))
   
    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size) 
        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        terminals = np.array([i[4] for i in batch])

        # Remover dimensões extras
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # Converter para tensores
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        
        # Obter valores Q para os próximos estados
        self.eval()
        with torch.no_grad():
            next_q_values = self(next_states_tensor)
        
        # Calcular valores Q alvo
        next_max = torch.max(next_q_values, dim=1)[0].cpu().numpy()
        targets = rewards + self.gamma * next_max * (1 - terminals)
        
        # Obter valores Q atuais
        self.eval()
        with torch.no_grad():
            targets_full = self(states_tensor).cpu().numpy()
        
        # Atualizar os valores Q das ações tomadas
        for i, action in enumerate(actions):
            targets_full[i][action] = targets[i]
        
        # Treinar a rede
        self.update(states, targets_full)
        
    def train_agent(self):
        rewards = []
        for i in range(self.episodes):

            state, _ = self.env.reset()  
        
            state = np.reshape(state, (1, self.state_dim))
            score = 0
            
            for t in range(self.timesteps):
                action = self.get_action(state)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                terminal = terminated or truncated
                
                score += reward
                next_state = np.reshape(next_state, (1, self.state_dim))
                self.add_exp(state, action, reward, next_state, terminal)
                state = next_state
                self.experience_replay()
                
                if terminal:
                    print(f'Episódio: {i+1}/{self.episodes}. Score: {score}. Epsilon: {self.epsilon}')
                    break
                    
            rewards.append(score)  
            
            # Decaimento do epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec
            
        return rewards


if __name__ == "__main__":
    env = gym.make('MountainCar-v0', render_mode=None)  
    agent = Agent(env, episodes=5000, timesteps=201, batch_size=64)
    rewards = agent.train_agent()
    np.save('learning_curve_data.npy', rewards)
    torch.save(agent.state_dict(), 'dqn_mountaincar-v.pth')