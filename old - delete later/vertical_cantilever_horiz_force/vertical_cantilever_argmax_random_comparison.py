# q_learning_testing.py

import numpy as np
import matplotlib.pyplot as plt
from vertical_cantilever_env import VerticalCantileverEnv  # Updated import
import time

# Q-learning parameters (not used here but kept for reference)
alpha = 0.1         # Learning rate
gamma = 0.99        # Discount factor
epsilon = 0.0       # Exploration rate set to 0 for testing (pure exploitation)
epsilon_decay = 0.999
epsilon_min = 0.01
num_test_episodes = 1
num_train_episodes = 1000
max_steps = 100

def run_test_episodes(env, Q_table, policy='argmax', num_episodes=5, max_steps=100):
    """
    Run test episodes under a specified policy.

    Args:
        env (gym.Env): The environment to run episodes in.
        Q_table (np.array): The Q-table containing learned Q-values.
        policy (str): 'argmax' for the learned policy or 'random' for random actions.
        num_episodes (int): Number of episodes to run.
        max_steps (int): Maximum steps per episode.

    Returns:
        list: List of total rewards per episode.
        list: List of force histories per episode.
        list: List of reward sequences per episode.
    """
    total_rewards = []
    all_force_histories = []
    all_rewards = []

    for episode in range(num_episodes):
        observation, _ = env.reset()
        state = env.discretize_observation(observation)
        rewards = []
        force_history = []
        total_reward = 0

        for step in range(max_steps):
            if policy == 'argmax':
                # Select the best action based on Q-table
                action = np.argmax(Q_table[state[0], state[1], :])
            elif policy == 'random':
                # Select a random action
                action = env.action_space.sample()
            else:
                raise ValueError("Policy must be either 'argmax' or 'random'.")

            # Execute the action
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = env.discretize_observation(next_observation)

            # Accumulate reward
            total_reward += reward
            rewards.append(reward)

            # Record force history
            force_history = env.force_history.copy()

            # Transition to next state
            state = next_state

            if terminated or truncated:
                break

        print(f"Policy: {policy.capitalize()} | Episode {episode+1}, Total Reward: {total_reward:.2f}")

        total_rewards.append(total_reward)
        all_force_histories.append(force_history)
        all_rewards.append(rewards)

        # Short pause to allow rendering
        if env.render_mode == 'human':
            time.sleep(0.5)  # Adjust as needed for visualization

    return total_rewards, all_force_histories, all_rewards

def plot_comparison(argmax_data, random_data, policy_labels=['Argmax Policy', 'Random Policy']):
    """
    Plot comparison graphs for both policies.

    Args:
        argmax_data (tuple): Tuple containing rewards, force histories, and reward sequences for argmax policy.
        random_data (tuple): Tuple containing rewards, force histories, and reward sequences for random policy.
        policy_labels (list): Labels for the policies in the plots.
    """
    argmax_rewards, argmax_force_histories, argmax_reward_sequences = argmax_data
    random_rewards, random_force_histories, random_reward_sequences = random_data

    # Plot Total Rewards Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(1, len(argmax_rewards)+1) - 0.2, argmax_rewards, width=0.4, label=policy_labels[0], color='blue')
    plt.bar(np.arange(1, len(random_rewards)+1) + 0.2, random_rewards, width=0.4, label=policy_labels[1], color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode: Argmax vs Random Policy')
    plt.legend()
    plt.xticks(np.arange(1, len(argmax_rewards)+1))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Plot Average Reward Comparison
    avg_argmax_reward = np.mean(argmax_rewards)
    avg_random_reward = np.mean(random_rewards)
    plt.figure(figsize=(8, 6))
    plt.bar(['Argmax Policy', 'Random Policy'], [avg_argmax_reward, avg_random_reward], color=['blue', 'orange'])
    plt.xlabel('Policy')
    plt.ylabel('Average Total Reward')
    plt.title('Average Total Reward: Argmax vs Random Policy')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Plot Force Histories for Each Episode
    for i in range(len(argmax_force_histories)):
        plt.figure(figsize=(10, 4))
        plt.plot(argmax_force_histories[i], label=policy_labels[0], color='blue')
        plt.plot(random_force_histories[i], label=policy_labels[1], color='orange')
        plt.title(f"Force History Comparison for Episode {i+1}")
        plt.xlabel("Time Step")
        plt.ylabel("Force (N)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Plot Reward vs Steps Comparison for Each Episode
    for i in range(len(argmax_reward_sequences)):
        plt.figure(figsize=(10, 4))
        plt.plot(argmax_reward_sequences[i], label=policy_labels[0], color='blue')
        plt.plot(random_reward_sequences[i], label=policy_labels[1], color='orange')
        plt.title(f"Reward vs. Steps Comparison for Episode {i+1}")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Initialize the environment without rendering for faster testing (change to 'human' if visualization is needed)
env = VerticalCantileverEnv(render_mode='human')

# Load the Q-table
Q_table = np.load(f'q_table_vertical_cantilever_{num_train_episodes}.npy')

# Run test episodes under Argmax Policy
print("Running test episodes under Argmax Policy...")
argmax_rewards, argmax_force_histories, argmax_reward_sequences = run_test_episodes(
    env, Q_table, policy='argmax', num_episodes=num_test_episodes, max_steps=max_steps
)

# Run test episodes under Random Policy
print("\nRunning test episodes under Random Policy...")
random_rewards, random_force_histories, random_reward_sequences = run_test_episodes(
    env, Q_table, policy='random', num_episodes=num_test_episodes, max_steps=max_steps
)

# Plot comparisons
plot_comparison(
    argmax_data=(argmax_rewards, argmax_force_histories, argmax_reward_sequences),
    random_data=(random_rewards, random_force_histories, random_reward_sequences),
    policy_labels=['Argmax Policy', 'Random Policy']
)

# Close the environment
env.close()
