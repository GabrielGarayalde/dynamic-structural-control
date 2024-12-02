import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

from vertical_cantilever_env import VerticalCantileverEnv
from dqn_agent import DQN  # Ensure this imports the correct DQN class

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the environment instance
env = VerticalCantileverEnv()

# Load the trained DQN models
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Define the ranges for displacement and velocity
num_displacement_bins = 100
num_velocity_bins = 100

displacement_values = np.linspace(env.min_displacement, env.max_displacement, num_displacement_bins)
velocity_values = np.linspace(env.min_velocity, env.max_velocity, num_velocity_bins)

# Get the list of saved model files
save_dir = 'saved_models_1000eps_400_freq4'
model_files = sorted([f for f in os.listdir(save_dir) if f.startswith('dqn_episode_') or f == 'dqn_final.pth'])

# Optionally, limit the number of models to plot
# model_files = model_files[::2]  # For example, plot every other model

# Generate heatmaps for each saved model
for model_file in model_files:
    model_path = os.path.join(save_dir, model_file)
    episode_number = model_file.split('_')[-1].split('.')[0] if 'episode' in model_file else 'Final'

    # Initialize the DQN model
    dqn_model = DQN(state_size, action_size).to(device)
    dqn_model.load_state_dict(torch.load(model_path, map_location=device))
    dqn_model.eval()

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

    # Plot the best action heatmap
    plt.figure(figsize=(10, 8))

    # Transpose the best_actions array to align displacement on x-axis and velocity on y-axis
    best_actions_transposed = best_actions.T

    # Define the extent of the heatmap
    extent = [displacement_values[0], displacement_values[-1], velocity_values[0], velocity_values[-1]]

    # Plot the heatmap
    plt.imshow(best_actions_transposed, extent=extent, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Best Action')

    plt.title(f"Best Action Heatmap at Episode {episode_number}")
    plt.xlabel("Displacement (m)")
    plt.ylabel("Velocity (m/s)")

    plt.tight_layout()
    plt.show()
