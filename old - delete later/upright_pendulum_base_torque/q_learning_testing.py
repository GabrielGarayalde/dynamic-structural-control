# q_learning_testing.py

import numpy as np
from discrete_pendulum_env import DiscretePendulumEnv
from discretization import discretize_observation
import matplotlib.pyplot as plt

# Load the Q-table
Q_table = np.load('q_table.npy')

max_steps = 100
num_test_episodes = 5

# Testing loop
env = DiscretePendulumEnv(render_mode='human')
for episode in range(num_test_episodes):
    observation, _ = env.reset()
    state = discretize_observation(observation)
    total_reward = 0
    for step in range(max_steps):
        action = np.argmax(Q_table[state[0], state[1], :])
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_observation(next_observation)
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
    print(f"Test Episode {episode+1}, Total Reward: {total_reward:.2f}")

    # Retrieve torque history
    torque_history = env.torque_history

    # Plot torque history
    plt.figure()
    plt.plot(torque_history)
    plt.title(f"Torque History for Episode {episode+1}")
    plt.xlabel("Time Step")
    plt.ylabel("Torque")
    plt.grid(True)
    plt.show()

env.close()
