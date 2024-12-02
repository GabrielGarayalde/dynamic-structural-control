# discrete_pendulum_env.py

from upright_pendulum_base_torque_env_dynamic import UprightPendulumEnv

from gymnasium.spaces import Discrete
from discretization import NUM_ACTIONS, discretize_action

class DiscretePendulumEnv(UprightPendulumEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.action_space = Discrete(NUM_ACTIONS)

    def step(self, action_index):
        # Convert action index to torque
        action = discretize_action(action_index)
        return super().step([action])
