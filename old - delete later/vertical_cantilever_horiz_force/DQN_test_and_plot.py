# dqn_testing.py

import numpy as np
import matplotlib.pyplot as plt
import torch
from vertical_cantilever_env import VerticalCantileverEnv
from dqn_agent import DQN  # Ensure this imports the correct DQN class

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set render mode
render_mode = 'human'  # Set to 'human' to visualize, or None for no rendering
render_mode = None  # Set to 'human' to visualize, or None for no rendering

# Create the environment
env = VerticalCantileverEnv(render_mode=render_mode)

# Initialize the DQN model
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

dqn_model = DQN(state_size, action_size).to(device)

# Load the trained DQN model
#dqn_model.load_state_dict(torch.load("dqn_vertical_cantilever.pth", map_location=device))
dqn_model.load_state_dict(torch.load("saved_models_500eps_800_freq0.5/dqn_final.pth", map_location=device))

dqn_model.eval()  # Set to evaluation mode

# Set epsilon to 0 for pure exploitation
epsilon = 0.0

# Testing parameters
num_test_episodes = 1
max_steps = 400  # Max steps per episode

# Testing loop
for episode in range(num_test_episodes):
    observation, _ = env.reset()
    state = observation  # No discretization needed for DQN
    rewards = []
    total_reward = 0
    done = False

    # Data storage
    force_history = []
    accel_ground_data = []
    time_data = []
    full_disp_data = []
    full_vel_data = []
    full_mass_accel_data = []

    for step in range(max_steps):
        # Select the best action based on the DQN model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = dqn_model(state_tensor)
            action = q_values.max(1)[1].item()
            #action = 7

        # Execute the action
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = next_observation

        # Accumulate reward
        total_reward += reward
        rewards.append(reward)

        # Collect data
        force_history.append(env.last_force)
        accel_ground_data.append(env.accel_ground_data[-1] if env.accel_ground_data else 0)
        full_mass_accel_data.append(env.full_mass_accel_data[-1] if env.full_mass_accel_data else 0)


        time_data.append(env.current_time)
        full_disp_data.append(env.state[0])
        full_vel_data.append(env.state[1])

        # Transition to next state
        state = next_state
        done = terminated or truncated

        if done:
            print("Episode terminated.")
            break

    print(f"Test Episode {episode+1}, Total Reward: {total_reward:.2f}")

    # Convert data to numpy arrays
    force_history = np.array(force_history)
    accel_ground_data = np.array(accel_ground_data)
    time_data = np.array(time_data)
    full_disp_data = np.array(full_disp_data)
    full_vel_data = np.array(full_vel_data)
    full_mass_accel_data = np.array(full_mass_accel_data)

    # Ensure arrays are the same length
    min_length = min(len(force_history), len(accel_ground_data), len(time_data))
    force_history = force_history[:min_length]
    accel_ground_data = accel_ground_data[:min_length]
    time_data = time_data[:min_length]
    full_disp_data = full_disp_data[:min_length]
    full_vel_data = full_vel_data[:min_length]
    full_mass_accel_data = full_mass_accel_data[:min_length]

    rewards = rewards[:min_length]

    # Plot Applied Force vs Ground Acceleration
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
    
    
    
    # Plotting with Twin Y-Axes
    plt.figure(figsize=(16, 8))
     
    # Create the first y-axis for Mass Acceleration
    ax1 = plt.gca()
    color1 = 'blue'
    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Mass Acceleration (m/s²)', color=color1, fontsize=14)
    ax1.plot(time_data, full_mass_accel_data, label='Mass Acceleration', color=color1, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
     
    # Center the first y-axis around zero
    accel_min = np.min(full_mass_accel_data)
    accel_max = np.max(full_mass_accel_data)
    accel_abs_max = max(abs(accel_min), abs(accel_max))
    ax1.set_ylim(-accel_abs_max * 1.1, accel_abs_max * 1.1)  # 10% padding
     
    # Create the second y-axis for Velocity
    ax2 = ax1.twinx()
    color2 = 'red'
    ax2.set_ylabel('Velocity (m/s)', color=color2, fontsize=14)
    ax2.plot(time_data, full_vel_data, label='Velocity', color=color2, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)
     
    # Center the second y-axis around zero
    vel_min = np.min(full_vel_data)
    vel_max = np.max(full_vel_data)
    vel_abs_max = max(abs(vel_min), abs(vel_max))
    ax2.set_ylim(-vel_abs_max * 1.1, vel_abs_max * 1.1)  # 10% padding
     
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12)
     
    # Add title and grid
    plt.title('Mass Acceleration and Velocity Over Time', fontsize=16)
    ax1.grid(True)
     
    # Ensure layout is tight so labels and titles are nicely spaced
    plt.tight_layout()
     
    # Display the plot
    plt.show()


    # Plot Displacement and Velocity over Time
    plt.figure(figsize=(16, 6))
    plt.plot(time_data, full_disp_data, label='Displacement', color='blue')
    plt.plot(time_data, full_vel_data, label='Velocity', color='green')
    plt.title(f"Displacement and Velocity over Time for Episode {episode+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Force History
    plt.figure(figsize=(16, 4))
    plt.plot(time_data, force_history, label='Applied Force', color='blue')
    plt.title(f"Force History for Episode {episode+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Reward vs Steps for the episode
    plt.figure(figsize=(16, 4))
    plt.plot(range(len(rewards)), rewards, label='Reward', color='green')
    plt.title(f"Reward vs. Steps for Episode {episode + 1}")
    plt.xlabel("Step")
    plt.ylabel("Reward")
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

env.close()
