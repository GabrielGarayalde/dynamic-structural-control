import numpy as np
import matplotlib.pyplot as plt
from box_mass_1d_env import BoxMass1DEnv
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter from PyTorch
import os
import datetime

# Q-learning parameters
alpha = 0.1         # Learning rate
gamma = 0.99        # Discount factor
epsilon = 1.0       # Exploration rate
epsilon_decay = 0.999
epsilon_min = 0.01
num_episodes = 1000
max_steps = 200

# Create the discrete environment
render_type = None
# render_type = 'human'

env = BoxMass1DEnv(render_mode=render_type)  # Set to 'human' to visualize

# Initialize Q-table
Q_table = np.full(
    (
        env.NUM_X_BINS,
        env.NUM_X_DOT_BINS,
        env.NUM_ACTIONS,
    ),
    -1.0  # Fill the Q-table with -1
)

# Initialize list to store total rewards
total_rewards = []

# Create a timestamp for the run
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Define a descriptive log directory
# (RL_env) C:\Users\gabri\Python>tensorboard --logdir=boxmassenv1d_tensorboard_runs/q_learning
log_dir = f"C:/Users/gabri/Python/boxmassenv1d_tensorboard_runs/q_learning/{timestamp}_num_episodes_{num_episodes}_max_steps_{max_steps}_freq_{env.freq}_eps_decay_{epsilon_decay}"
os.makedirs(log_dir, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir)

# Define hyperparameters and metrics dictionaries
hparam_dict = {
    'num_episodes': num_episodes,
    'max_steps': max_steps,
    'epsilon_decay': epsilon_decay,
    'freq_harmonic': env.freq
}

metric_dict = {
    'total_reward': 0,        # Initial metric values
    'average_td_error': 0,
    'average_q_value': 0
}

# Log hyperparameters with initial metrics
writer.add_hparams(hparam_dict, metric_dict)

# Training loop
for episode in range(num_episodes):
    observation, _ = env.reset()
    state = env.discretize_observation(observation)
    total_reward = 0
    total_td_error = 0  # For logging average TD error per episode
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
        total_td_error += abs(td_error)  # Accumulate absolute TD error
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
    # Append total reward of the episode
    total_rewards.append(total_reward)

    # Calculate metrics
    average_td_error = total_td_error / (step + 1)
    average_q_value = np.mean(Q_table)

    # Log metrics to TensorBoard
    writer.add_scalar('Total Reward', total_reward, episode)
    writer.add_scalar('Epsilon', epsilon, episode)
    writer.add_scalar('Average TD Error', average_td_error, episode)
    writer.add_scalar('Average Q-Value', average_q_value, episode)

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Optional: Print progress
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
        print(f"Average TD Error: {average_td_error:.4f}, Average Q-Value: {average_q_value:.4f}")

# Log final metrics associated with hyperparameters
final_metric_dict = {
    'total_reward': np.mean(total_rewards[-100:]),  # Example: average of last 100 episodes
    'average_td_error': average_td_error,
    'average_q_value': average_q_value
}
writer.add_hparams(hparam_dict, final_metric_dict)

# Close the environment and writer
env.close()
writer.close()

# Save the Q-table
np.save(f"q_table_{num_episodes}_freq2.npy", Q_table)

# Calculate average reward every 100 episodes
window_size = 100
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
