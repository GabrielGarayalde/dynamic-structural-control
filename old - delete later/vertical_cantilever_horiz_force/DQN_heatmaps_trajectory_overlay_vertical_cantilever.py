import numpy as np
import matplotlib.pyplot as plt
import torch

from vertical_cantilever_env import VerticalCantileverEnv
from dqn_agent import DQN  # Ensure this imports the correct DQN class

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the environment instance
env = VerticalCantileverEnv(render_mode=None)  # Set to 'human' if you want to render the environment

# Load the trained DQN model
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the DQN model
dqn_model = DQN(state_size, action_size).to(device)
dqn_model.load_state_dict(torch.load("dqn_vertical_cantilever_500.pth", map_location=device))
dqn_model.load_state_dict(torch.load("saved_models_1000eps_400_freq0.5/dqn_episode_1000.pth", map_location=device))

dqn_model.eval()

# Define the ranges for displacement and velocity
num_displacement_bins = 100
num_velocity_bins = 100

displacement_values = np.linspace(env.min_displacement, env.max_displacement, num_displacement_bins)

displacement_values = np.linspace(-0.024, 0.024, num_displacement_bins)
velocity_values = np.linspace(-0.15, .15, num_velocity_bins)

# Initialize an array to hold Q-values
Q_values = np.zeros((num_displacement_bins, num_velocity_bins, action_size))

# Compute Q-values for each state in the grid
with torch.no_grad():
    for i, x in enumerate(displacement_values):
        for j, x_dot in enumerate(velocity_values):
            # Prepare the state
            state = np.array([x, x_dot], dtype=np.float32)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            # Get Q-values from the model
            q_values = dqn_model(state_tensor).cpu().numpy().flatten()
            Q_values[i, j, :] = q_values

# Find the best action at each state
best_actions = np.argmax(Q_values, axis=2)  # Shape: (num_displacement_bins, num_velocity_bins)

# Run a test episode and collect the state trajectory
observation, _ = env.reset()
state = observation
trajectory_displacements = []
trajectory_velocities = []
done = False

for _ in range(200):  # Max steps per episode
    # Store the state
    trajectory_displacements.append(state[0])
    trajectory_velocities.append(state[1])
    
    # Select the best action using the trained DQN model
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = dqn_model(state_tensor)
        action = q_values.max(1)[1].item()
        #action = 7

    
    # Execute the action
    next_observation, reward, terminated, truncated, _ = env.step(action)
    next_state = next_observation
    state = next_state
    done = terminated or truncated
    if done:
        break

# Plot the best action heatmap with the trajectory overlaid
plt.figure(figsize=(12, 8))

# Transpose the best_actions array to align displacement on x-axis and velocity on y-axis
best_actions_transposed = best_actions.T

# Define the extent of the heatmap
extent = [displacement_values[0], displacement_values[-1], velocity_values[0], velocity_values[-1]]

# Plot the heatmap
plt.imshow(best_actions_transposed, extent=extent, origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Best Action')

plt.title("Best Action Heatmap with Trajectory Overlay")
plt.xlabel("Displacement (m)")
plt.ylabel("Velocity (m/s)")

# Overlay the trajectory
plt.plot(trajectory_displacements, trajectory_velocities, color='red', linewidth=2, label='Agent Trajectory')

# Optionally, plot starting and ending points
plt.scatter(trajectory_displacements[0], trajectory_velocities[0], color='green', s=100, marker='o', label='Start')
plt.scatter(trajectory_displacements[-1], trajectory_velocities[-1], color='white', s=100, marker='X', label='End')

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
