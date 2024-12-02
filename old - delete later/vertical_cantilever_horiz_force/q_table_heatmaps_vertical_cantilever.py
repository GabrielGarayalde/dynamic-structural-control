import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from vertical_cantilever_env import VerticalCantileverEnv  # Ensure your environment is correctly imported

# Create the environment instance
env = VerticalCantileverEnv()

# Define the number of training episodes (ensure this matches your saved Q-table)
num_training_eps = 5000

# Define the number of steps for the test episode
max_steps = 400

# Example: Define the actual displacement and velocity ranges based on the environment
displacement_values = np.linspace(env.min_displacement, env.max_displacement, env.NUM_X_BINS)
velocity_values = np.linspace(env.min_velocity, env.max_velocity, env.NUM_X_DOT_BINS)

# Load the Q-table (ensure the file exists)
Q_table = np.load(f'q_table_{num_training_eps}_freq2.npy')

# Select the middle values for displacement bins (disregard the first and last bins)
middle_Q_table = Q_table[1:-1, :, :]  # Shape: (num_displacement_bins - 2, num_velocity_bins, num_actions)

# Define the corresponding displacement values for the truncated bins
middle_displacement_values = displacement_values[1:-1]

# Find the action with the highest Q-value for each state in the middle Q-table
best_actions = np.argmax(middle_Q_table, axis=2)  # Shape: (num_displacement_bins - 2, num_velocity_bins)

# -------------------------------
# Step 1: Swap Axes (Displacement on X, Velocity on Y)
# -------------------------------

# Transpose the best_actions array to swap axes
best_actions_transposed = best_actions.T  # Shape: (num_velocity_bins, num_displacement_bins - 2)

# -------------------------------
# Step 2: Invert the Velocity Axis
# -------------------------------

# Reverse the order of velocity values for the y-axis
velocity_values_inverted = velocity_values[::-1]

# Reverse the rows in the transposed best_actions array to match the inverted velocity axis
best_actions_transposed_inverted = best_actions_transposed[::-1, :]  # Shape remains (num_velocity_bins, num_displacement_bins - 2)

# -------------------------------
# Step 3: Plot the Heatmap with Swapped and Inverted Axes
# -------------------------------

plt.figure(figsize=(12, 8))

# Create the heatmap
ax = sns.heatmap(
    best_actions_transposed_inverted,
    cmap='viridis',
    annot=True,
    fmt='d',
    cbar=True,
    xticklabels=np.round(middle_displacement_values, 2),
    yticklabels=np.round(velocity_values_inverted, 2)
)

# Set plot titles and labels
ax.set_title("Best Action Heatmap with Trajectory (Displacement vs. Velocity)", fontsize=16)
ax.set_xlabel("Displacement (m)", fontsize=14)
ax.set_ylabel("Velocity (m/s)", fontsize=14)

# -------------------------------
# Step 4: Run Test Episode and Collect Trajectory Data
# -------------------------------

# Initialize variables to collect trajectory data
displacement_history = []
velocity_history = []

# Run a test episode
observation, _ = env.reset()
state = env.discretize_observation(observation)
total_reward = 0

for step in range(max_steps):
    # Select the best action based on Q-table
    action = np.argmax(Q_table[state[0], state[1], :])
    
    # Execute the action
    next_observation, reward, terminated, truncated, _ = env.step(action)
    next_state = env.discretize_observation(next_observation)
    
    # Accumulate reward
    total_reward += reward
    
    # Collect displacement and velocity for trajectory
    displacement_history.append(next_observation[0])  # Continuous value
    velocity_history.append(next_observation[1])      # Continuous value
    
    # Transition to next state
    state = next_state
    
    if terminated or truncated:
        print(f"Test episode terminated at step {step+1}.")
        break

print(f"Test Episode, Total Reward: {total_reward:.2f}")

# -------------------------------
# Step 5: Overlay Trajectory on Heatmap
# -------------------------------

# Map the trajectory data to indices corresponding to the heatmap's axes

# For displacement (x-axis), map displacement_history to indices of middle_displacement_values
displacement_indices = np.interp(
    displacement_history,
    middle_displacement_values,
    np.arange(len(middle_displacement_values))
)

# For velocity (y-axis), map velocity_history to indices of velocity_values
# Use the original velocity_values (not inverted) because they are increasing
velocity_indices = np.interp(
    velocity_history,
    velocity_values,
    np.arange(len(velocity_values))
)

# Since the heatmap y-axis is inverted, adjust the velocity indices
velocity_indices_inverted = len(velocity_values) - 1 - velocity_indices

# Adjust displacement indices to match the heatmap's x-axis (if necessary)
# In this case, middle_displacement_values is increasing, matching the heatmap's x-axis

# Plot the trajectory over the heatmap
ax.plot(
    displacement_indices,
    velocity_indices_inverted,
    marker='o',
    markersize=4,
    color='red',
    linewidth=2,
    label='Trajectory'
)

# Add legend
ax.legend(loc='upper right', facecolor='white', framealpha=0.8)

# Improve layout
plt.tight_layout()

# -------------------------------
# Step 6: Show or Save the Plot
# -------------------------------

# Optionally save the plot
plots_dir = 'best_action_heatmaps_with_trajectory'
os.makedirs(plots_dir, exist_ok=True)
plot_filename = f"best_action_heatmap_with_trajectory_eps_{num_training_eps}.png"
plot_path = os.path.join(plots_dir, plot_filename)
plt.savefig(plot_path, dpi=300)

print(f"Saved Best Action Heatmap with Trajectory at {plot_path}")

# Display the plot interactively (if running in an environment that supports it)
plt.show()

# Close the plot to free memory
plt.close()
