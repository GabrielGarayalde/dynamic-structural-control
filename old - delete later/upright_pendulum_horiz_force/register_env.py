# register_env.py

from gymnasium.envs.registration import register

register(
    id='DiscreteUprightPendulum-v0',
    entry_point='discrete_pendulum_env:DiscretePendulumEnv',
)
