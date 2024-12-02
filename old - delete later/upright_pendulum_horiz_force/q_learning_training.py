import numpy as np
import matplotlib.pyplot as plt

from discrete_pendulum_env import DiscretePendulumEnv

# Q-learning parameters
alpha = 0.1         # Learning rate
gamma = 0.99        # Discount factor
epsilon = 1.0       # Exploration rate
epsilon_decay = 0.999
epsilon_min = 0.01
num_episodes = 5000
max_steps = 200

# Specify the continuous environment module name
continuous_env_module_name = 'upright_pendulum_horiz_force_env_dynamic'  # Or any other module

# Create the discrete environment
env = DiscretePendulumEnv(continuous_env_module_name)

# Initialize Q-table
Q_table = np.zeros(
    (
        env.NUM_THETA_BINS,
        env.NUM_THETA_DOT_BINS,
        env.NUM_ACTIONS,
    )
)

# Initialize list to store total rewards
total_rewards = []

# Training loop
for episode in range(num_episodes):
    observation, _ = env.reset()
    state = env.discretize_observation(observation)
    total_reward = 0
    for step in range(max_steps):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state[0], state[1], :])
        # Execute action
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = env.discretize_observation(next_observation)
        # Update Q-table
        best_next_action = np.argmax(Q_table[next_state[0], next_state[1], :])
        td_target = reward + gamma * Q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - Q_table[state[0], state[1], action]
        Q_table[state[0], state[1], action] += alpha * td_error
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
    # Append total reward of the episode
    total_rewards.append(total_reward)
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    # Optional: Print progress
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
env.close()

# Save the Q-table
np.save('q_table.npy', Q_table)

# Calculate average reward every 10 episodes
window_size = 20  # You can change this to any number of episodes you want to average over
average_rewards = [
    np.mean(total_rewards[i:i + window_size])
    for i in range(0, len(total_rewards), window_size)
]

# Create corresponding episode numbers for plotting
episode_numbers = np.arange(len(average_rewards)) * window_size

# Plot average rewards vs episodes
plt.figure(figsize=(10, 5))
plt.plot(episode_numbers, average_rewards, marker='o')
plt.xlabel('Episode')
plt.ylabel(f'Average Total Reward (per {window_size} episodes)')
plt.title('Average Rewards vs Episodes')
plt.grid(True)
plt.show()
