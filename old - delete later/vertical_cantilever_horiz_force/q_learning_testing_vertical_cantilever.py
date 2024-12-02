# q_learning_testing.py

import numpy as np
import matplotlib.pyplot as plt
from vertical_cantilever_env import VerticalCantileverEnv  # Updated import

# Q-learning parameters
alpha = 0.1         # Learning rate
gamma = 0.99        # Discount factor
epsilon = 0.0       # Exploration rate set to 0 for testing (pure exploitation)
epsilon_decay = 0.999
epsilon_min = 0.01
num_test_episodes = 1
max_steps = 400

num_training_eps = 5000

render_mode = 'human'
render_mode = None

# Create the environment with rendering enabled
env = VerticalCantileverEnv(render_mode=render_mode)

# Load the Q-table
Q_table = np.load(f'q_table_{num_training_eps}_freq2.npy')

# Testing loop
for episode in range(num_test_episodes):
    observation, _ = env.reset()
    state = env.discretize_observation(observation)
    rewards = []
    total_reward = 0
    for step in range(max_steps):
        # Select the best action based on Q-table
        action = np.argmax(Q_table[state[0], state[1], :])
        
        # Select a random action (actions are 0, 1, 2, 3, 4)
        #action = env.action_space.sample()  # Random action selection
        #action = 7
        # Execute the action
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = env.discretize_observation(next_observation)
        
        # Accumulate reward
        total_reward += reward
        rewards.append(reward)
        
        # Transition to next state
        state = next_state
        
        if terminated or truncated:
            print("terminated")
            break
    
    print(f"Test Episode {episode+1}, Total Reward: {total_reward:.2f}")

    # After the episode ends
    force_history = np.array(env.full_force_history)
    accel_ground_data = np.array(env.full_accel_ground_data)
    time_data = np.array(env.full_time_data)
    full_disp_data = np.array(env.full_disp_data)
    full_vel_data = np.array(env.full_vel_data)
    
    # Ensure arrays are the same length
    min_length = min(len(force_history), len(accel_ground_data), len(time_data))
    force_history = force_history[:min_length]
    accel_ground_data = accel_ground_data[:min_length]
    time_data = time_data[:min_length]
    
    # Plot force history and ground acceleration overlay
    plt.figure(figsize=(16, 6))
    plt.plot(time_data, force_history, label='Applied Force', color='blue', linewidth=2)
    plt.plot(time_data, accel_ground_data, label='Ground Acceleration', color='red', linestyle='--', linewidth=2)
    
    plt.title(f"Applied Force vs. Ground Acceleration for Episode {episode+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N) / Acceleration (m/s²)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()





    # Create figure and primary axis for displacement
    fig, ax1 = plt.subplots(figsize=(16, 4))
    
    # Plot displacement on the primary y-axis (left)
    ax1.plot(force_history, label='Force', color='red')
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Force", color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)
    
    # Create a secondary y-axis for velocity, also on the left side
    ax2 = ax1.twinx()  # Create a twin axis sharing the same x-axis
    ax2.plot(full_disp_data, label='Displacement', color='blue')
    ax2.set_ylabel("Displacement", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.spines['left'].set_visible(True)  # Ensure left spine is visible
    
    # Create a third y-axis for force history on the right side
    ax3 = ax1.twinx()  # Create a twin y-axis for force data
    ax3.spines['right'].set_position(('outward', 60))  # Shift the third y-axis outward
    ax3.plot(full_vel_data, label='Velocity', color='green')
    ax3.set_ylabel("Velocity", color='green')
    ax3.tick_params(axis='y', labelcolor='green')
    
    # Ensure zero-centered limits for each variable
    # Force limits
    force_min, force_max = min(force_history), max(force_history)
    force_limit = max(abs(force_min), abs(force_max))  # Symmetric around 0
    ax1.set_ylim(-force_limit*1.1, force_limit*1.1)
    
    # Displacement limits
    disp_min, disp_max = min(full_disp_data), max(full_disp_data)
    disp_limit = max(abs(disp_min), abs(disp_max))  # Symmetric around 0
    ax2.set_ylim(-disp_limit*1.1, disp_limit*1.1)
    
    # Velocity limits
    vel_min, vel_max = min(full_vel_data), max(full_vel_data)
    vel_limit = max(abs(vel_min), abs(vel_max))  # Symmetric around 0
    ax3.set_ylim(-vel_limit*1.1, vel_limit*1.1)
    
    
    # Set the title and layout
    plt.title(f"Displacement, Velocity, and Force History for Episode {episode+1}")
    fig.tight_layout()
    
    plt.show()



    # Plot force history
    plt.figure(figsize=(10, 4))
    plt.plot(force_history, label='Applied Force', color='blue')
    plt.title(f"Force History for Episode {episode+1}")
    plt.xlabel("Time Step")
    plt.ylabel("Force (N)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
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
