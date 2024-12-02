import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class UprightPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.max_speed = 8.0
        self.max_force = 10.0  # Maximum force magnitude
        self.theta_min = -np.pi / 4
        self.theta_max = np.pi / 4
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.dynamics_dt = 0.05  # Dynamics time step
        self.action_dt = 0.2     # Action time step
        self.num_integration_steps = int(self.action_dt / self.dynamics_dt)
        self.fps = int(1 / self.dynamics_dt)  # Frames per second based on dynamics_dt

        # Discretize the force range
        self.NUM_ACTIONS = 5  # Number of discrete actions
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.force_bins = np.linspace(-self.max_force, self.max_force, self.NUM_ACTIONS)

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
        self.last_force = 0.0  # Keep track of the last force applied

        # Font for displaying text
        self.font = None
        self.start_time = None  # For timer

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        high = np.array([0.05, 0.05])  # Small perturbation
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_force = 0.0  # Reset last force

        # Reset start time for timer
        if self.render_mode == "human":
            self.start_time = pygame.time.get_ticks()
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        theta, theta_dot = self.state
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        # Map action index to force value
        force = self.force_bins[action]
        self.last_force = force

        terminated = False  # Initialize the terminated flag

        for _ in range(self.num_integration_steps):
            theta, theta_dot = self.state

            # Torque due to gravity
            tau_gravity = self.m * self.g * self.l * np.sin(theta)

            # Torque due to applied force
            tau_force = force * self.l * np.cos(theta)

            # Total torque
            tau_total = tau_gravity + tau_force

            # Angular acceleration
            theta_ddot = tau_total / (self.m * self.l**2)

            # Update state
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
            pygame.display.set_caption("Upright Pendulum with Cables")
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
        origin = (250, 400)  # Adjusted to place pivot lower
        length_pixels = 200  # pixels
        x_m = origin[0] + length_pixels * np.sin(theta)
        y_m = origin[1] - length_pixels * np.cos(theta)

        # Ground attachment points
        left_ground = (origin[0] - length_pixels, origin[1])
        right_ground = (origin[0] + length_pixels, origin[1])

        # Draw the pendulum rod
        pygame.draw.line(self.screen, (0, 0, 0), origin, (x_m, y_m), 5)
        pygame.draw.circle(self.screen, (0, 0, 255), (int(x_m), int(y_m)), 15)

        # Calculate apparent tensions for visualization
        # For simplicity, assume tensions are proportional to the net force and cable geometry
        # Not physically accurate, but for visualization purposes
        # Calculate cable lengths and unit vectors
        delta_x_left = x_m - left_ground[0]
        delta_y_left = y_m - left_ground[1]
        length_left = np.hypot(delta_x_left, delta_y_left)
        u_left_x = delta_x_left / length_left
        u_left_y = delta_y_left / length_left

        delta_x_right = x_m - right_ground[0]
        delta_y_right = y_m - right_ground[1]
        length_right = np.hypot(delta_x_right, delta_y_right)
        u_right_x = delta_x_right / length_right
        u_right_y = delta_y_right / length_right

        # Compute tensions (for visualization only)
        T_left = self.last_force / 2
        T_right = self.last_force / 2

        # Draw the cables
        # Left cable
        color_left = (200, 200, 200)
        pygame.draw.line(self.screen, color_left, (x_m, y_m), left_ground, 3)

        # Right cable
        color_right = (200, 200, 200)
        pygame.draw.line(self.screen, color_right, (x_m, y_m), right_ground, 3)

        # Draw the force arrow (horizontal force)
        force = self.last_force
        max_force = self.max_force

        # Scale the force for visualization
        force_scale = 50
        arrow_length = (force / max_force) * force_scale

        # Arrow direction
        arrow_start = (x_m, y_m)
        arrow_end = (x_m + arrow_length, y_m)  # Horizontal arrow

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
        self.screen.blit(time_surface, (350, 10))  # Position at top right

        # Update the display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
