import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import gymnasium as gym

# Import your VerticalCantileverEnv
from vertical_cantilever_env import VerticalCantileverEnv

# DQN Neural Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.99         # Discount factor
        self.epsilon = 1.0        # Exploration rate
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.buffer_size = 100000
        self.update_target_every = 1000  # Update target network every n steps
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(self.buffer_size)
        self.steps_done = 0
    
    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
    
    def push_memory(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        
        # Compute loss
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main training loop
if __name__ == "__main__":
    # Initialize environment
    env = VerticalCantileverEnv(render_mode=None)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    num_episodes = 500
    max_steps = 800  # Max steps per episode
    total_rewards = []
    
    # Create a directory to save models
    import os
    save_dir = 'saved_models_500eps_800_freq0.5/'
    os.makedirs(save_dir, exist_ok=True)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        for step in range(max_steps):
            # Select and perform an action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store the transition in memory
            agent.push_memory(state, action, reward, next_state, done)
            
            # Move to the next state
            state = next_state
            total_reward += reward
            
            # Perform one step of the optimization
            agent.train()
            
            if done:
                break
        
        total_rewards.append(total_reward)
        
        # Save the model every 50 episodes
        if (episode + 1) % 50 == 0:
            model_path = os.path.join(save_dir, f"dqn_episode_{episode+1}.pth")
            torch.save(agent.policy_net.state_dict(), model_path)
            print(f"Model saved at episode {episode+1}")
        
        # Optional: Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    # Save the final trained model
    torch.save(agent.policy_net.state_dict(), os.path.join(save_dir, "dqn_final.pth"))
    
    # Plot the rewards
    import matplotlib.pyplot as plt
    
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()
    
    env.close()
