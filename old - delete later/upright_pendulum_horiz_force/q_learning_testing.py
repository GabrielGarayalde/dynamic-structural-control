# q_learning_testing.py

import numpy as np
import matplotlib.pyplot as plt
from discrete_pendulum_env import DiscretePendulumEnv

# Load the Q-table
Q_table = np.load('q_table.npy')

max_steps = 100
num_test_episodes = 5

# Specify the continuous environment module name (same as used during training)
continuous_env_module_name = 'upright_pendulum_horiz_force_env_dynamic'  # Replace with your module name if different

# Initialize the environment
env = DiscretePendulumEnv(continuous_env_module_name, render_mode='human')

# Testing loop
for episode in range(num_test_episodes):
    observation, _ = env.reset()
    state = env.discretize_observation(observation)
    rewards = []
    total_reward = 0
    for step in range(max_steps):
        action = np.argmax(Q_table[state[0], state[1], :])
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = env.discretize_observation(next_observation)
        total_reward += reward
        
        # Append the current reward to rewards list
        rewards.append(reward)
        
        state = next_state
        if terminated or truncated:
            break
    print(f"Test Episode {episode+1}, Total Reward: {total_reward:.2f}")

    # Retrieve force history from the continuous environment
    force_history = env.env.force_history  # Accessing force_history from the wrapped continuous environment

    # Plot force history
    plt.figure()
    plt.plot(force_history)
    plt.title(f"Force History for Episode {episode+1}")
    plt.xlabel("Time Step")
    plt.ylabel("Force")
    plt.grid(True)
    plt.show()
    
    # Plot Reward vs Steps for the episode
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label='Reward', color='green')
    plt.title(f"Reward vs. Steps for Episode {episode + 1}")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

env.close()
