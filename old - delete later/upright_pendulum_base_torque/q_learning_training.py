# q_learning_training.py

import numpy as np
from discrete_pendulum_env import DiscretePendulumEnv
from discretization import (
    NUM_THETA_BINS, NUM_THETA_DOT_BINS, NUM_ACTIONS,
    discretize_observation
)

# Q-learning parameters
alpha = 0.1         # Learning rate
gamma = 0.99        # Discount factor
epsilon = 1.0       # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 500
max_steps = 200

# Initialize Q-table
Q_table = np.zeros((NUM_THETA_BINS, NUM_THETA_DOT_BINS, NUM_ACTIONS))

# Training loop
env = DiscretePendulumEnv()
for episode in range(num_episodes):
    observation, _ = env.reset()
    state = discretize_observation(observation)
    total_reward = 0
    for step in range(max_steps):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state[0], state[1], :])
        # Execute action
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_observation(next_observation)
        # Update Q-table
        best_next_action = np.argmax(Q_table[next_state[0], next_state[1], :])
        td_target = reward + gamma * Q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - Q_table[state[0], state[1], action]
        Q_table[state[0], state[1], action] += alpha * td_error
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    # Optional: Print progress
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
env.close()

# Save the Q-table
np.save('q_table.npy', Q_table)
