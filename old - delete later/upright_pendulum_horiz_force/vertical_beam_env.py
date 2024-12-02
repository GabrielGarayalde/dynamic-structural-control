import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class VerticalBeamEnv(gym.Env):
    """
    A Gym environment simulating a vertical beam fixed to the ground with a lumped mass at the top.
    The agent applies horizontal forces to the mass to minimize its horizontal displacement,
    accounting for ground motion disturbances modeled as white noise.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        # ----- Physical Parameters -----
        self.m = 1.0              # Mass (kg)
        self.E = 200e9            # Elastic Modulus (Pa) - Example: Steel
        self.I = 1e-4             # Area Moment of Inertia (m^4) - Example value
        self.l = 1.0              # Length of the beam (meters)
        self.k = 3 * self.E * self.I / self.l**3  # Stiffness (N/m) for a cantilever beam
        self.c = 2.0              # Damping coefficient (Ns/m)
        self.g = 9.81             # Acceleration due to gravity (m/s^2)

        # ----- Simulation Parameters -----
        self.dynamics_dt = 0.05   # Time step for dynamics (seconds)
        self.action_dt = 0.2      # Time step for actions (seconds)
        self.num_integration_steps = int(self.action_dt / self.dynamics_dt)
        self.fps = int(1 / self.dynamics_dt)  # Frames per second based on dynamics_dt

        # ----- Action Space -----
        self.NUM_ACTIONS = 5
        self.max_force = 10.0     # Maximum horizontal force (N)
        self.force_bins = np.linspace(-self.max_force, self.max_force, self.NUM_ACTIONS)
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)

        # ----- Observation Space -----
        # State: [x, x_dot]
        self.x_limit = 2.0        # Maximum horizontal displacement (meters)
        self.v_limit = 10.0       # Maximum horizontal velocity (m/s)
        self.observation_space = spaces.Box(
            low=np.array([-self.x_limit, -self.v_limit], dtype=np.float32),
            high=np.array([self.x_limit, self.v_limit], dtype=np.float32),
            dtype=np.float32,
        )

        # ----- Rendering -----
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.start_time = None  # For timer

        # ----- State Initialization -----
        self.state = None
        self.np_random = np.random.default_rng()

        # ----- Force Tracking -----
        self.last_force = 0.0

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns an initial observation.

        Args:
            seed (int, optional): Seed for the environment's random number generator.
            options (dict, optional): Additional options for resetting the environment.

        Returns:
            observation (numpy.ndarray): The initial observation of the environment.
            info (dict): Additional info (empty in this case).
        """
        super().reset(seed=seed)

        # Initialize state with small random displacement and zero velocity
        high = np.array([0.05, 0.05], dtype=np.float32)  # Small perturbation
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_force = 0.0  # Reset last force

        # Reset start time for timer
        if self.render_mode == "human":
            self.start_time = pygame.time.get_ticks()
            self.render()

        return self._get_obs(), {}

    def _get_obs(self):
        """
        Returns the current observation of the environment.

        Returns:
            observation (numpy.ndarray): Current state [x, x_dot].
        """
        x, x_dot = self.state
        return np.array([x, x_dot], dtype=np.float32)

    def step(self, action):
        """
        Executes one time step within the environment.

        Args:
            action (int): The action to take.

        Returns:
            observation (numpy.ndarray): The next observation.
            reward (float): The reward for this step.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information (empty in this case).
        """
        assert self.action_space.contains(action), "Invalid Action"

        # Map action index to force value
        force = self.force_bins[action]
        self.last_force = force

        # Generate ground acceleration as white noise (earthquake effect)
        sigma = 0.5  # Standard deviation for ground acceleration (m/s^2)
        a_g = self.np_random.normal(0, sigma)

        terminated = False  # Initialize termination flag

        for _ in range(self.num_integration_steps):
            x, x_dot = self.state

            # Compute horizontal acceleration based on equation of motion:
            # m * x'' + c * x' + k * x = F + m * a_g
            x_ddot = (force + self.m * a_g - self.c * x_dot - self.k * x) / self.m

            # Update velocity and position using Euler integration
            x_dot += x_ddot * self.dynamics_dt
            x_dot = np.clip(x_dot, -self.v_limit, self.v_limit)
            x += x_dot * self.dynamics_dt
            x = np.clip(x, -self.x_limit, self.x_limit)

            # Update state
            self.state = np.array([x, x_dot], dtype=np.float32)

            # Check termination condition
            if abs(x) >= self.x_limit:
                terminated = True
                break  # Exit the integration loop early

            # Render intermediate steps if rendering is enabled
            if self.render_mode == "human":
                self.render()

        # Calculate reward
        x, x_dot = self.state
        # Reward: Negative squared displacement to minimize it
        # Optional: Penalize force usage for efficiency
        force_penalty = 0.001 * (force ** 2)  # Adjust the weight as needed
        reward = - (x ** 2) - force_penalty

        truncated = False  # Placeholder for possible truncation logic

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        """
        Renders the current state of the environment.
        """
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((600, 600))
            pygame.display.set_caption("Vertical Beam with Lumped Mass")
            self.font = pygame.font.SysFont('Arial', 18)
            self.start_time = pygame.time.get_ticks()  # Initialize start time

        if self.clock is None:
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((255, 255, 255))  # White background

        # Get the current state
        x, _ = self.state

        # Convert physical coordinates to screen coordinates
        origin = (300, 100)  # Fixed pivot point on the screen
        scale = 100          # Pixels per meter

        # Calculate mass position
        mass_x = origin[0] + x * scale
        mass_y = origin[1] + self.l * scale  # Fixed beam length

        # Draw the beam (a vertical line from origin to mass)
        pygame.draw.line(self.screen, (0, 0, 0), origin, (mass_x, mass_y), 5)

        # Draw the lumped mass as a circle
        pygame.draw.circle(self.screen, (0, 0, 255), (int(mass_x), int(mass_y)), 20)

        # Draw the applied force as an arrow
        force = self.last_force
        max_force = self.max_force

        # Scale the force for visualization
        force_scale = 50  # Pixels per unit force
        arrow_length = (force / max_force) * force_scale

        # Arrow direction: positive force to the right, negative to the left
        arrow_start = (mass_x, mass_y)
        arrow_end = (mass_x + arrow_length, mass_y)

        # Choose color based on force direction
        arrow_color = (255, 0, 0) if force > 0 else (0, 255, 0)

        # Draw the force arrow
        pygame.draw.line(self.screen, arrow_color, arrow_start, arrow_end, 5)

        # Draw arrowhead
        if force != 0:
            arrowhead_size = 10
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
        force_text = f"Force: {force:.2f} N"
        text_surface = self.font.render(force_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        # Display elapsed time
        elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000  # Convert to seconds
        time_text = f"Time: {elapsed_time:.2f}s"
        time_surface = self.font.render(time_text, True, (0, 0, 0))
        self.screen.blit(time_surface, (450, 10))  # Position at top right

        # Update the display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        """
        Closes the rendering window and quits Pygame.
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
