# discretization.py

import numpy as np

# Discretization parameters
NUM_THETA_BINS = 15
NUM_THETA_DOT_BINS = 15
NUM_ACTIONS = 20

# Define the ranges
theta_min, theta_max = -np.pi, np.pi
theta_dot_min, theta_dot_max = -8.0, 8.0  # Based on environment's max_speed
action_min, action_max = -1.0, 1.0  # Torque limits

# Create bins
theta_bins = np.linspace(theta_min, theta_max, NUM_THETA_BINS)
theta_dot_bins = np.linspace(theta_dot_min, theta_dot_max, NUM_THETA_DOT_BINS)
action_bins = np.linspace(action_min, action_max, NUM_ACTIONS)

def discretize_observation(observation):
    cos_theta, sin_theta, theta_dot = observation
    theta = np.arctan2(sin_theta, cos_theta)
    theta_bin = np.digitize(theta, theta_bins) - 1
    theta_bin = np.clip(theta_bin, 0, NUM_THETA_BINS - 1)
    theta_dot_bin = np.digitize(theta_dot, theta_dot_bins) - 1
    theta_dot_bin = np.clip(theta_dot_bin, 0, NUM_THETA_DOT_BINS - 1)
    return theta_bin, theta_dot_bin

def discretize_action(action_index):
    return action_bins[action_index]
