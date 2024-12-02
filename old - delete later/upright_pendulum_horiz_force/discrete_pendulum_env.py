# discrete_pendulum_env.py

import numpy as np
import importlib
from gymnasium.spaces import Discrete

class DiscretePendulumEnv:
    # Discretization parameters
    NUM_THETA_BINS = 20          # Number of bins for theta
    NUM_THETA_DOT_BINS = 20      # Number of bins for theta_dot
    NUM_ACTIONS = 20              # Number of discrete actions


    def __init__(self, continuous_env_module_name, render_mode=None):
        # Dynamically import the continuous environment module
        env_module = importlib.import_module(continuous_env_module_name)
        # Get the continuous environment class (assumed to be named 'UprightPendulumEnv')
        continuous_env_class = getattr(env_module, 'UprightPendulumEnv')
        # Create an instance of the continuous environment
        self.env = continuous_env_class(render_mode=render_mode)

        # Set MAX_SPEED and MAX_FORCE from the continuous environment
        self.MAX_SPEED = self.env.max_speed
        self.MAX_FORCE = self.env.max_force
        self.THETA_MIN = self.env.theta_min
        self.THETA_MAX = self.env.theta_max

        # Generate bins for discretization
        self.theta_bins = np.linspace(self.THETA_MIN, self.THETA_MAX, self.NUM_THETA_BINS)
        self.theta_dot_bins = np.linspace(-self.MAX_SPEED, self.MAX_SPEED, self.NUM_THETA_DOT_BINS)
        self.action_bins = np.linspace(-self.MAX_FORCE, self.MAX_FORCE, self.NUM_ACTIONS)

        # Discrete action space
        self.action_space = Discrete(self.NUM_ACTIONS)
        # Observation space remains the same as the continuous environment
        self.observation_space = self.env.observation_space

    def step(self, action_index):
        # Convert action index to continuous action
        action = self.discretize_action(action_index)
        # Execute the action in the continuous environment
        obs, reward, terminated, truncated, info = self.env.step([action])
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def discretize_observation(self, observation):
        cos_theta, sin_theta, theta_dot = observation
        theta = np.arctan2(sin_theta, cos_theta)
        theta_bin = np.digitize(theta, self.theta_bins) - 1
        theta_bin = np.clip(theta_bin, 0, self.NUM_THETA_BINS - 1)
        theta_dot_bin = np.digitize(theta_dot, self.theta_dot_bins) - 1
        theta_dot_bin = np.clip(theta_dot_bin, 0, self.NUM_THETA_DOT_BINS - 1)
        return theta_bin, theta_dot_bin

    def discretize_action(self, action_index):
        return self.action_bins[action_index]
