import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class UprightPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.max_speed = 8.0
        self.max_force = 10.0
        self.theta_min = -np.pi / 4
        self.theta_max = np.pi / 4
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.dynamics_dt = 0.05  # Dynamics time step
        self.action_dt = 0.2     # Action time step
        self.num_integration_steps = int(self.action_dt / self.dynamics_dt)
        self.fps = int(1 / self.dynamics_dt)  # Frames per second based on dynamics_dt

        # Action space represents the horizontal force F
        self.action_space = spaces.Box(
            low=-self.max_force,
            high=self.max_force,
            shape=(1,),
            dtype=np.float32,
        )

        # Observation space: [cos(theta), sin(theta), theta_dot]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -self.max_speed]),
            high=np.array([1.0, 1.0, self.max_speed]),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.state = None
        self.np_random = np.random.default_rng()

        # Initialize last force
        self.last_force = 0.0

        # Font for displaying text
        self.font = None
        self.start_time = None  # For timer
        self.force_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        high = np.array([0.1, 0.1])  # Small perturbation
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_force = 0.0  # Reset last force
        self.force_history = []  # Initialize force history

        # Reset start time for timer
        if self.render_mode == "human":
            self.start_time = pygame.time.get_ticks()
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        theta, theta_dot = self.state
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def step(self, action):
        force = np.clip(action, -self.max_force, self.max_force)[0]
        self.last_force = force
        self.force_history.append(force)

        terminated = False  # Initialize the terminated flag

        for _ in range(self.num_integration_steps):
            theta, theta_dot = self.state

            # Dynamics equations using the smaller dynamics_dt
            theta_ddot = (self.g / self.l) * np.sin(theta) + (force / (self.m * self.l)) * np.cos(theta)

            theta_dot += theta_ddot * self.dynamics_dt
            theta_dot = np.clip(theta_dot, -self.max_speed, self.max_speed)
            theta += theta_dot * self.dynamics_dt

            # Check if theta is out of bounds
            if theta < self.theta_min or theta > self.theta_max:
                terminated = True
                break  # Exit the integration loop early

            self.state = np.array([theta, theta_dot])

            # Render intermediate steps
            if self.render_mode == "human":
                self.render()

        # If the episode was terminated inside the loop, update the state accordingly
        if terminated:
            self.state = np.array([theta, theta_dot])

        # Calculate reward
        theta, theta_dot = self.state
        uprightness = np.cos(theta)
        reward = uprightness

        truncated = False  # You can set this based on a time limit if needed

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((500, 500))
            pygame.display.set_caption("Upright Pendulum with External Force")
            self.font = pygame.font.SysFont('Arial', 18)
            self.start_time = pygame.time.get_ticks()  # Initialize start time
        if self.clock is None:
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((255, 255, 255))

        # Get the current angle
        theta, _ = self.state

        # Pendulum parameters
        origin = (250, 250)
        length = 200  # pixels
        pendulum_x = origin[0] + length * np.sin(theta)
        pendulum_y = origin[1] - length * np.cos(theta)

        # Draw the pendulum
        pygame.draw.line(self.screen, (0, 0, 0), origin, (pendulum_x, pendulum_y), 5)
        pygame.draw.circle(self.screen, (0, 0, 255), (int(pendulum_x), int(pendulum_y)), 15)

        # Draw the force arrow at the pendulum bob
        force = self.last_force
        max_force = self.max_force

        # Scale the force for visualization
        force_scale = 50
        arrow_length = (force / max_force) * force_scale

        # Arrow direction
        arrow_start = (pendulum_x, pendulum_y)
        arrow_end = (pendulum_x + arrow_length, pendulum_y)  # Horizontal arrow

        # Draw the arrow
        arrow_color = (255, 0, 0) if force > 0 else (0, 255, 0)
        pygame.draw.line(self.screen, arrow_color, arrow_start, arrow_end, 5)

        # Draw arrowhead
        arrowhead_size = 10
        if force != 0:
            angle = 0 if force > 0 else np.pi
            arrow_tip = arrow_end
            left_wing = (
                arrow_tip[0] + arrowhead_size * np.cos(angle + np.pi / 4),
                arrow_tip[1] + arrowhead_size * np.sin(angle + np.pi / 4),
            )
            right_wing = (
                arrow_tip[0] + arrowhead_size * np.cos(angle - np.pi / 4),
                arrow_tip[1] + arrowhead_size * np.sin(angle - np.pi / 4),
            )
            pygame.draw.polygon(self.screen, arrow_color, [arrow_tip, left_wing, right_wing])

        # Display force magnitude as text
        force_text = f"Force: {force:.2f}"
        text_surface = self.font.render(force_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        # Display elapsed time
        elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000  # Convert to seconds
        time_text = f"Time: {elapsed_time:.2f}s"
        time_surface = self.font.render(time_text, True, (0, 0, 0))
        self.screen.blit(time_surface, (400, 10))  # Position at top right

        # Update the display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
