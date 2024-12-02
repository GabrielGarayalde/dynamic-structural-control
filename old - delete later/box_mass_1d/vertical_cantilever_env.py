# vertical_cantilever_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random  # For white noise generation
import sys
import math
from collections import deque

class VerticalCantileverEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # ----------------------------
        # Environment Parameters
        # ----------------------------
        self.max_displacement = 0.01  # Maximum horizontal displacement (meters)
        self.max_velocity = 0.1     # Maximum horizontal velocity (m/s)
        self.max_force = 1.0        # Maximum force magnitude (Newtons)
        self.min_displacement = -self.max_displacement # Minimum horizontal displacement (meters)
        self.min_velocity = -self.max_velocity      # Minimum horizontal velocity (m/s)

        # Physical constants
        self.E = 2.0e9
        self.I = 25
        self.L = 1         # Length of the cantilever in meters (m)
        self.k = 3.0 * self.E * self.I / (self.L*1000)**3.0
        
        # Overriding stiffness for testing purposes
        self.k = (2*math.pi)**2
        self.m = 1.0     # Mass (kg)
        
        
        self.freq_n = 1.0           # Natural frequency (Hz)
        self.omega_n = self.freq_n * 2 * math.pi  # Natural angular frequency (rad/s)
        self.k = self.omega_n ** 2 * self.m    # Stiffness (N/m)
        
        # Time steps
        self.current_time = 0.0
        self.dynamics_dt = 0.001 * 2  # Physics update time step (seconds)
        self.action_dt = self.dynamics_dt * 10    # Agent action time step (seconds)
        self.num_integration_steps = int(self.action_dt / self.dynamics_dt)
        self.fps = int(1 / self.dynamics_dt)  # Frames per second based on dynamics_dt

        # State space: displacemnt and velocity
        self.NUM_X_BINS = 15
        self.NUM_X_DOT_BINS = 15

        # Action space: Discrete forces
        self.NUM_ACTIONS = 15  # Number of discrete actions
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.force_bins = np.linspace(-self.max_force, self.max_force, self.NUM_ACTIONS)

        # Observation space: [x, x_dot]
        self.observation_space = spaces.Box(
            low=np.array([self.min_displacement, self.min_velocity]),
            high=np.array([self.max_displacement, self.max_velocity]),
            dtype=np.float32,
        )

        # Rendering and Plotting Setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Scaling and Visualization Parameters
        self.WIDTH, self.HEIGHT = 1200, 1200  # Increased height to accommodate four plots
        self.scale_displacement = 400          # pixels per meter for mass displacement
        self.scale_length = 500                # pixels per meter for vertical beam length
        self.base_x_initial = self.WIDTH // 4
        self.base_y = 3 * self.HEIGHT // 4  # Positioned higher to accommodate plots below
        self.mass_radius = 10
        self.N_segments = 20  # Number of segments to discretize the beam
        self.z_values = [i * self.L / self.N_segments for i in range(self.N_segments + 1)]  # Positions along the beam

        # Plot properties
        self.T_max = 10.0   # Cutoff time for plotting in seconds
        self.plot_width = self.WIDTH - self.base_x_initial - 300
        self.plot_height = (self.HEIGHT - 200) / 4 - 50  # Adjusted for four plots
        self.plot_x = self.base_x_initial + 200
        self.plot_y_disp = 100
        self.plot_y_vel = self.plot_y_disp + self.plot_height + 50
        self.plot_y_mass_accel = self.plot_y_vel + self.plot_height + 50
        self.plot_y_accel_ground = self.plot_y_mass_accel + self.plot_height + 50

        # Data buffers for rendering (limited length)
        self.max_points = int(self.T_max * self.fps)
        self.disp_data = deque(maxlen=self.max_points)
        self.vel_data = deque(maxlen=self.max_points)
        self.mass_accel_data = deque(maxlen=self.max_points)    # Buffer for mass acceleration
        self.accel_ground_data = deque(maxlen=self.max_points)  # Buffer for ground acceleration
        self.time_data = deque(maxlen=self.max_points)

        # Initialize data buffers with initial conditions
        for _ in range(self.max_points):
            self.disp_data.append(0.0)            # Initial displacement
            self.vel_data.append(0.0)             # Initial velocity
            self.mass_accel_data.append(0.0)      # Initial mass acceleration
            self.accel_ground_data.append(0.0)    # Initial ground acceleration
            self.time_data.append(-self.T_max + _ * self.dynamics_dt)

        # Full-length data buffers for the entire episode
        self.full_disp_data = []
        self.full_vel_data = []
        self.full_mass_accel_data = []
        self.full_accel_ground_data = []
        self.full_time_data = []
        self.full_force_history = []  # Full force history at dynamics_dt intervals

        # Initial state
        self.state = None
        self.np_random = np.random.default_rng()

        # Initialize last force
        self.last_force = 0.0  # Keep track of the last force applied
        self.force_history = []  # Force history at dynamics_dt intervals
        self.action_force_history = []  # Force history at action_dt intervals

        # Font for displaying text
        self.font = None
        self.small_font = None
        self.start_time = None  # For timer

        # Initialize rendering if required
        if self.render_mode == "human":
            self._initialize_rendering()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # high = np.array([0.05, 0.05])  # Small initial displacement and velocity
        # self.state = self.np_random.uniform(low=-high, high=high)
        
        self.state = [0,0] # static initial config
        self.last_force = 0.0  # Reset last force
        self.force_history = []  # Reset force history
        self.action_force_history = []  # Reset action force history
        self.current_time = 0.0

        # Reset data buffers for rendering
        self.disp_data.clear()
        self.vel_data.clear()
        self.mass_accel_data.clear()
        self.accel_ground_data.clear()
        self.time_data.clear()

        # Initialize data buffers with initial conditions for rendering
        for _ in range(self.max_points):
            self.disp_data.append(0.0)
            self.vel_data.append(0.0)
            self.mass_accel_data.append(0.0)
            self.accel_ground_data.append(0.0)
            self.time_data.append(-self.T_max + _ * self.dynamics_dt)

        # Reset full-length data buffers for the episode
        self.full_disp_data = []
        self.full_vel_data = []
        self.full_mass_accel_data = []
        self.full_accel_ground_data = []
        self.full_time_data = []
        self.full_force_history = []

        # Reset start time for timer
        if self.render_mode == "human":
            if self.screen is None:
                self._initialize_rendering()
            self.start_time = pygame.time.get_ticks()

        return self._get_obs(), {}

    def _get_obs(self):
        x, x_dot = self.state
        return np.array([x, x_dot], dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        # Map action index to force value
        force = self.force_bins[action]
        self.last_force = force
        self.action_force_history.append(force)  # Record force at action step

        terminated = False  # Initialize the terminated flag
        truncated = False   # Initialize the truncated flag

        for _ in range(self.num_integration_steps):
            x, x_dot = self.state

            # Generate white noise for ground acceleration
            a_white_noise = random.uniform(-1, 1) * 1  # White noise in the range [-10, 10] m/s²

            # Parameters for sinusoidal ground acceleration
            amplitude = 1  # Amplitude of the sinusoidal ground acceleration (m/s²)
            freq = 0.5             # Excitation frequency (Hz)
            omega = freq * 2 * math.pi  # Excitation angular frequency (rad/s)

            # Update current time
            self.current_time += self.dynamics_dt

            # Generate sinusoidal ground acceleration
            a_sinusoidal = amplitude * math.sin(omega * self.current_time)  # Sinusoidal acceleration

            # Total ground acceleration (sum of white noise and sinusoidal component)
            a_ground = a_sinusoidal  # You can add a_white_noise if desired

            # Physics integration using Euler method
            # x'' = (-k/m) * x + (F/m) + a_ground
            x_ddot = (-self.k / self.m) * x + (force / self.m) - a_ground
            # x_ddot = (force / self.m) - a_ground

            # Update velocity and displacement
            x_dot += x_ddot * self.dynamics_dt
            #x_dot = np.clip(x_dot, self.min_velocity, self.max_velocity)
            x += x_dot * self.dynamics_dt

            # Update state
            self.state = np.array([x, x_dot])

            # Update data buffers for rendering
            self.force_history.append(force)  # Record the force at dynamics_dt intervals
            self.disp_data.append(x)
            self.vel_data.append(x_dot)
            self.mass_accel_data.append(x_ddot)
            self.accel_ground_data.append(a_ground)
            self.time_data.append(self.current_time)

            # Update full-length data buffers for the episode
            self.full_force_history.append(force)
            self.full_disp_data.append(x)
            self.full_vel_data.append(x_dot)
            self.full_mass_accel_data.append(x_ddot)
            self.full_accel_ground_data.append(a_ground)
            self.full_time_data.append(self.current_time)

            # Check if displacement is out of bounds
            # if x < self.min_displacement or x > self.max_displacement:
            #     print("Displacement is out of bounds")
                # terminated = True
                # break  # Exit the integration loop early

            # Render intermediate steps
            if self.render_mode == "human":
                self._render_visualization()

        # Calculate reward
        # Reward is negative absolute displacement to minimize it
        reward = -abs(self.state[0])

        # Optionally, you can add penalties for high velocity or forces

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return
        if self.screen is None:
            self._initialize_rendering()
        self._render_visualization()

    def _initialize_rendering(self):
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Vertical Cantilever Simulation with Detailed Plots")
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 18)
        self.clock = pygame.time.Clock()

    def _render_visualization(self):
        # Handle events to prevent Pygame from freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Clear screen
        self.screen.fill((255, 255, 255))  # White background

        # Extract state variables
        x, x_dot = self.state

        # Draw fixed base
        pygame.draw.circle(self.screen, (0, 0, 0), (self.base_x_initial, self.base_y), 10)

        # Calculate mass position (fixed vertically with deflection)
        mass_x = self.base_x_initial + x * self.scale_displacement
        mass_y = self.base_y - self.L * self.scale_length  # Mass is at the end of the beam vertically

        # Draw the beam based on displacement
        self._draw_beam(x, mass_x, mass_y)

        # Draw mass as a red circle
        pygame.draw.circle(self.screen, (255, 0, 0), (int(mass_x), int(mass_y)), self.mass_radius)

        # Draw force arrow
        self._draw_force_arrow(mass_x, mass_y)

        # Draw text information
        if self.start_time:
            current_time = (pygame.time.get_ticks() - self.start_time) / 1000  # seconds
        else:
            current_time = 0.0
        info_text = f"Time: {current_time:.2f} s | Displacement: {x:.3f} m | Velocity: {x_dot:.3f} m/s | Force: {self.last_force:.2f} N"
        text_surface = self.font.render(info_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        # Draw Displacement Plot with dynamic scaling and dynamic y-axis labels
        self._draw_plot(self.disp_data, self.plot_x, self.plot_y_disp, self.plot_width, self.plot_height,
                       (0, 0, 255), "Displacement", "m")

        # Draw Velocity Plot with dynamic scaling and dynamic y-axis labels
        self._draw_plot(self.vel_data, self.plot_x, self.plot_y_vel, self.plot_width, self.plot_height,
                       (0, 200, 0), "Velocity", "m/s")

        # Draw Mass Acceleration Plot with dynamic scaling and dynamic y-axis labels
        self._draw_plot(self.mass_accel_data, self.plot_x, self.plot_y_mass_accel, self.plot_width, self.plot_height,
                       (105, 105, 105), "Mass Acceleration", "m/s²")

        # Draw Ground Acceleration Plot with dynamic scaling and dynamic y-axis labels
        self._draw_plot(self.accel_ground_data, self.plot_x, self.plot_y_accel_ground, self.plot_width, self.plot_height,
                       (255, 0, 0), "Ground Acceleration", "m/s²")

        # Titles for plots
        disp_title = self.small_font.render(f"Displacement vs Time (Last {self.T_max} s)", True, (0, 0, 0))
        self.screen.blit(disp_title, (self.plot_x, self.plot_y_disp + self.plot_height + 5))
        vel_title = self.small_font.render(f"Velocity vs Time (Last {self.T_max} s)", True, (0, 0, 0))
        self.screen.blit(vel_title, (self.plot_x, self.plot_y_vel + self.plot_height + 5))
        mass_accel_title = self.small_font.render(f"Mass Acceleration vs Time (Last {self.T_max} s)", True, (0, 0, 0))
        self.screen.blit(mass_accel_title, (self.plot_x, self.plot_y_mass_accel + self.plot_height + 5))
        accel_ground_title = self.small_font.render(f"Ground Acceleration vs Time (Last {self.T_max} s)", True, (0, 0, 0))
        self.screen.blit(accel_ground_title, (self.plot_x, self.plot_y_accel_ground + self.plot_height + 5))

        # Update the display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_beam(self, x_displacement, mass_x, mass_y):
        # Helper function to draw the beam based on discretized points
        beam_points = []
        for z in self.z_values:
            # Calculate deflection based on mode shape (quadratic)
            phi = (z / self.L) ** 2
            x_deflection = x_displacement * phi
            # Map to screen coordinates
            beam_x = self.base_x_initial + x_deflection * self.scale_displacement  # Horizontal displacement
            beam_y = self.base_y - z * self.scale_length  # Vertical position
            beam_points.append((beam_x, beam_y))
        pygame.draw.lines(self.screen, (0, 0, 0), False, beam_points, 4)

    def _draw_force_arrow(self, mass_x, mass_y):
        # Draw the force arrow applied to the mass
        force = self.last_force
        max_force = self.max_force
        force_scale = 50  # Scale for visualization
        arrow_length = (force / max_force) * force_scale
        arrow_start = (mass_x, mass_y)
        arrow_end = (mass_x + arrow_length, mass_y)

        # Determine arrow color based on force direction
        arrow_color = (0, 255, 0) if force >= 0 else (255, 0, 0)

        # Draw the arrow line
        pygame.draw.line(self.screen, arrow_color, arrow_start, arrow_end, 5)
        # Draw arrowhead
        if force != 0:
            angle = 0 if force > 0 else math.pi
            arrow_tip = arrow_end
            wing_length = 10
            wing_angle = math.pi / 6  # 30 degrees
            left_wing = (
                arrow_tip[0] - wing_length * math.cos(angle - wing_angle),
                arrow_tip[1] - wing_length * math.sin(angle - wing_angle),
            )
            right_wing = (
                arrow_tip[0] - wing_length * math.cos(angle + wing_angle),
                arrow_tip[1] - wing_length * math.sin(angle + wing_angle),
            )
            pygame.draw.polygon(self.screen, arrow_color, [arrow_tip, left_wing, right_wing])

        # Display force magnitude as text
        force_text = f"Force: {force:.2f} N"
        text_surface = self.small_font.render(force_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (mass_x + arrow_length + 10, mass_y - 10))

    def _draw_plot(self, data, plot_x, plot_y, plot_width, plot_height, color, label, y_label):
        # Draw individual plots with dynamic scaling and dynamic y-axis labels
        pygame.draw.rect(self.screen, (220, 220, 220), (plot_x, plot_y, plot_width, plot_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (plot_x, plot_y, plot_width, plot_height), 2)
        # Axes
        pygame.draw.line(self.screen, (0, 0, 0), (plot_x, plot_y + plot_height//2),
                         (plot_x + plot_width, plot_y + plot_height//2), 1)  # X-axis
        pygame.draw.line(self.screen, (0, 0, 0), (plot_x, plot_y),
                         (plot_x, plot_y + plot_height), 1)  # Y-axis

        # Determine the maximum absolute value in the data for scaling
        max_val = max(abs(val) for val in data) if data else 1.0
        nice_scale = self._get_nice_scale(max_val)

        # Calculate scaling factor based on nice_scale
        padding = 10  # pixels
        scale = (plot_height / 2 - padding) / nice_scale if nice_scale != 0 else 1

        # Plot data
        if len(data) > 1:
            points = []
            for i in range(len(data)):
                t = self.time_data[i]
                value = data[i]
                # Only plot data within the cutoff time window
                if t < (self.time_data[-1] - self.T_max):
                    continue
                # Map time to x-coordinate
                x_coord = plot_x + (t - (self.time_data[-1] - self.T_max)) / self.T_max * plot_width
                # Map value to y-coordinate
                y_coord = plot_y + plot_height//2 - value * scale
                points.append((x_coord, y_coord))
            if points:
                pygame.draw.lines(self.screen, color, False, points, 2)

        # Labels
        label_surface = self.small_font.render(label, True, (0, 0, 0))
        self.screen.blit(label_surface, (plot_x, plot_y - 30))
        
        # Y-axis labels
        # Center label (0)
        pygame.draw.line(self.screen, (0, 0, 0), (plot_x - 5, plot_y + plot_height//2 - 1),
                         (plot_x + 5, plot_y + plot_height//2 -1), 1)  # Center
        zero_label = self.small_font.render("0", True, (0, 0, 0))
        self.screen.blit(zero_label, (plot_x - 20, plot_y + plot_height//2 - 10))
        
        # Positive label
        pygame.draw.line(self.screen, (169, 169, 169), (plot_x - 5, plot_y + plot_height//2 - nice_scale * scale),
                         (plot_x + 5, plot_y + plot_height//2 - nice_scale * scale), 1)  # Positive
        pos_label = self.small_font.render(f"{nice_scale:.2f} {y_label}", True, (0, 0, 0))
        self.screen.blit(pos_label, (plot_x - 60, plot_y + plot_height//2 - nice_scale * scale - 10))
        
        # Negative label
        pygame.draw.line(self.screen, (169, 169, 169), (plot_x - 5, plot_y + plot_height//2 + nice_scale * scale),
                         (plot_x + 5, plot_y + plot_height//2 + nice_scale * scale), 1)  # Negative
        neg_label = self.small_font.render(f"-{nice_scale:.2f} {y_label}", True, (0, 0, 0))
        self.screen.blit(neg_label, (plot_x - 70, plot_y + plot_height//2 + nice_scale * scale - 10))

        # Optional: Draw grid lines for better readability
        # Positive grid line
        pygame.draw.line(self.screen, (169, 169, 169), (plot_x, plot_y + plot_height//2 - nice_scale * scale),
                         (plot_x + plot_width, plot_y + plot_height//2 - nice_scale * scale), 1)
        # Negative grid line
        pygame.draw.line(self.screen, (169, 169, 169), (plot_x, plot_y + plot_height//2 + nice_scale * scale),
                         (plot_x + plot_width, plot_y + plot_height//2 + nice_scale * scale), 1)

    def _get_nice_scale(self, max_val):
        """
        Calculate a "nice" scale for the y-axis labels.
        """
        if max_val == 0:
            return 1  # Avoid log10(0)
        magnitude = 10 ** math.floor(math.log10(max_val))
        residual = max_val / magnitude
        if residual <= 1:
            nice_scale = magnitude
        elif residual <= 2:
            nice_scale = 2 * magnitude
        elif residual <= 5:
            nice_scale = 5 * magnitude
        else:
            nice_scale = 10 * magnitude
        return nice_scale

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    def discretize_observation(self, observation):
        """
        Discretize the continuous observation into discrete bins for Q-learning.
        Args:
            observation (np.array): Continuous observation [x, x_dot]
        Returns:
            tuple: Discrete state indices (x_bin, x_dot_bin)
        """
        x, x_dot = observation

        # Define number of bins
        

        # Define bin edges
        x_bins = np.linspace(self.min_displacement, self.max_displacement, self.NUM_X_BINS + 1)
        x_dot_bins = np.linspace(self.min_velocity, self.max_velocity, self.NUM_X_DOT_BINS + 1)

        # Digitize the observations
        x_bin = np.digitize(x, x_bins) - 1
        x_dot_bin = np.digitize(x_dot, x_dot_bins) - 1

        # Handle edge cases
        x_bin = np.clip(x_bin, 0, self.NUM_X_BINS - 1)
        x_dot_bin = np.clip(x_dot_bin, 0, self.NUM_X_DOT_BINS - 1)

        return (x_bin, x_dot_bin)
