import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from vertical_cantilever_env import VerticalCantileverEnv
from dqn_agent import DQN  # Ensure this imports the correct DQN class

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the environment instance
env = VerticalCantileverEnv()

# Load the trained DQN model
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the DQN model
dqn_model = DQN(state_size, action_size).to(device)
dqn_model.load_state_dict(torch.load("dqn_vertical_cantilever.pth", map_location=device))
dqn_model.eval()

# Define the ranges for displacement and velocity
num_displacement_bins = 50
num_velocity_bins = 50

displacement_values = np.linspace(env.min_displacement, env.max_displacement, num_displacement_bins)
velocity_values = np.linspace(env.min_velocity, env.max_velocity, num_velocity_bins)

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

# Optionally, map action indices to force values
force_bins = env.force_bins  # Assuming this is available

# # Generate heatmaps for each action
# for action in range(action_size):
#     q_values_for_action = Q_values[:, :, action]  # Shape: (num_displacement_bins, num_velocity_bins)
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(q_values_for_action, cmap='coolwarm', annot=False, cbar=True,
#                 xticklabels=np.round(velocity_values, 3),
#                 yticklabels=np.round(displacement_values, 3))
#     plt.title(f"Q-Value Heatmap for Action {action} (Force: {force_bins[action]:.2f} N)")
#     plt.xlabel("Velocity (m/s)")
#     plt.ylabel("Displacement (m)")
#     plt.xticks(rotation=45)
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.show()

# Find the best action at each state
best_actions = np.argmax(Q_values, axis=2)  # Shape: (num_displacement_bins, num_velocity_bins)

# Create a heatmap for the best actions
plt.figure(figsize=(10, 8))
sns.heatmap(best_actions, cmap='viridis', annot=False, cbar=True,
            xticklabels=np.round(velocity_values, 3),
            yticklabels=np.round(displacement_values, 3))
plt.title("Best Action Heatmap (Argmax of Q-Values)")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Displacement (m)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
