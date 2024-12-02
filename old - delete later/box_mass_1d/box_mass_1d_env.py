# vertical_cantilever_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random  # For white noise generation
import sys
import math
from collections import deque

class BoxMass1DEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, enable_plots=False):
        super().__init__()
        
        # ----------------------------
        # Environment Parameters
        # ----------------------------
        self.max_displacement = 0.01  # Maximum horizontal displacement (meters)
        self.max_velocity = 0.1       # Maximum horizontal velocity (m/s)
        self.max_force = 1.0          # Maximum force magnitude (Newtons)
        self.min_displacement = -self.max_displacement  # Minimum horizontal displacement (meters)
        self.min_velocity = -self.max_velocity          # Minimum horizontal velocity (m/s)

        # Physical constants
        self.m = 1.0     # Mass (kg)
        self.c = 0.1     # Damping coefficient (Ns/m)
        
        self.freq_n = 1.0           # Natural frequency (Hz)
        self.omega_n = self.freq_n * 2 * math.pi  # Natural angular frequency (rad/s)
        self.k = self.omega_n ** 2 * self.m    # Stiffness (N/m)
        
        self.freq = 2
        # self.k = 50.0   # Spring stiffness (N/m)
        
        
        
        # Time steps
        self.current_time = 0.0
        self.dynamics_dt = 0.001       # Physics update time step (seconds)
        self.num_integration_steps = 10
        self.action_dt = self.dynamics_dt * self.num_integration_steps    # Agent action time step (seconds)
        self.fps = 60  # Fixed FPS for rendering

        # State space: displacement and velocity
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

        # Enable or disable plotting and data storage
        self.enable_plots = enable_plots

        # Scaling and Visualization Parameters
        self.WIDTH, self.HEIGHT = 1400, 1000  # Adjusted to accommodate two boxes and plots
        self.scale_displacement = 200          # pixels per meter for mass displacement
        self.box_width = 200
        self.box_height = 200
        self.box_top = 200
        self.mass_radius = 10

        # Positions for the renderings
        self.first_box_left = 50
        self.second_box_left = self.first_box_left + self.box_width + 100  # 100 pixels between boxes

        # Plot properties
        self.T_max = 10.0   # Cutoff time for plotting in seconds
        self.plot_x = self.second_box_left + self.box_width + 150
        self.plot_width = 600  # Made plots narrower
        self.plot_height = (self.HEIGHT - 150) / 5  # For five plots
        
        self.plot_padding = 60  # Increased padding from 30 to 50 pixels

        self.plot_y_disp = 100
        self.plot_y_vel = self.plot_y_disp + self.plot_height + self.plot_padding
        self.plot_y_mass_accel = self.plot_y_vel + self.plot_height + self.plot_padding
        self.plot_y_accel_ground = self.plot_y_mass_accel + self.plot_height + self.plot_padding
        self.plot_y_force = self.plot_y_accel_ground + self.plot_height + self.plot_padding  # Position for force plot


        self.max_points = int(self.T_max * (1 / self.dynamics_dt))  # Max points based on dynamics_dt

        # Data buffers for rendering (limited length)
        if self.enable_plots:
            self.disp_data = deque(maxlen=self.max_points)
            self.vel_data = deque(maxlen=self.max_points)
            self.mass_accel_data = deque(maxlen=self.max_points)    # Buffer for mass acceleration
            self.accel_ground_data = deque(maxlen=self.max_points)  # Buffer for ground acceleration
            self.force_data = deque(maxlen=self.max_points)         # Buffer for force data
            self.time_data = deque(maxlen=self.max_points)

            # Initialize data buffers with initial conditions
            for i in range(self.max_points):
                self.disp_data.append(0.0)            # Initial displacement
                self.vel_data.append(0.0)             # Initial velocity
                self.mass_accel_data.append(0.0)      # Initial mass acceleration
                self.accel_ground_data.append(0.0)    # Initial ground acceleration
                self.force_data.append(0.0)           # Initial force
                self.time_data.append(-self.T_max + i * self.dynamics_dt)

            # Full-length data buffers for the entire episode
            self.full_disp_data = []
            self.full_vel_data = []
            self.full_mass_accel_data = []
            self.full_accel_ground_data = []
            self.full_force_history = []
            self.full_time_data = []
        else:
            # If plots are disabled, no need to initialize data buffers
            pass

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

        # Base (box) position and velocity
        self.x_base = 0.0
        self.v_base = 0.0

        # Initialize rendering if required
        if self.render_mode == "human":
            self._initialize_rendering()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.state = [0.0, 0.0]  # Static initial configuration
        self.last_force = 0.0  # Reset last force
        self.force_history = []  # Reset force history
        self.action_force_history = []  # Reset action force history
        self.current_time = 0.0

        # Reset base position and velocity
        self.x_base = 0.0
        self.v_base = 0.0

        if self.enable_plots:
            # Reset data buffers for rendering
            self.disp_data.clear()
            self.vel_data.clear()
            self.mass_accel_data.clear()
            self.accel_ground_data.clear()
            self.force_data.clear()
            self.time_data.clear()

            # Initialize data buffers with initial conditions for rendering
            for i in range(self.max_points):
                self.disp_data.append(0.0)
                self.vel_data.append(0.0)
                self.mass_accel_data.append(0.0)
                self.accel_ground_data.append(0.0)
                self.force_data.append(0.0)
                self.time_data.append(-self.T_max + i * self.dynamics_dt)

            # Reset full-length data buffers for the episode
            self.full_disp_data = []
            self.full_vel_data = []
            self.full_mass_accel_data = []
            self.full_accel_ground_data = []
            self.full_force_history = []
            self.full_time_data = []
        else:
            # If plots are disabled, no need to reset data buffers
            pass

        # Reset start time for timer
        if self.render_mode == "human":
            if self.screen is None:
                self._initialize_rendering()
            self.start_time = pygame.time.get_ticks()

        return self._get_obs(), {}

    def _get_obs(self):
        x_rel, x_dot_rel = self.state
        return np.array([x_rel, x_dot_rel], dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        # Map action index to force value
        force = self.force_bins[action]
        self.last_force = force
        self.action_force_history.append(force)  # Record force at action step

        terminated = False  # Initialize the terminated flag
        truncated = False   # Initialize the truncated flag

        for _ in range(self.num_integration_steps):
            x_rel, x_dot_rel = self.state

            # Generate white noise for ground acceleration
            a_white_noise = random.uniform(-1, 1) * 1  # White noise in the range [-1, 1] m/s²

            # Parameters for sinusoidal ground acceleration
            amplitude = 1  # Amplitude of the sinusoidal ground acceleration (m/s²)
            freq = self.freq     # Excitation frequency (Hz)
            omega = freq * 2 * math.pi  # Excitation angular frequency (rad/s)

            # Update current time
            self.current_time += self.dynamics_dt

            # Generate sinusoidal ground acceleration
            a_sinusoidal = amplitude * math.sin(omega * self.current_time)  # Sinusoidal acceleration

            # Total ground acceleration (sum of white noise and sinusoidal component)
            a_ground = a_sinusoidal  # You can add a_white_noise if desired

            # Update base (box) velocity and position
            a_base = a_ground
            self.v_base += a_base * self.dynamics_dt
            self.x_base += self.v_base * self.dynamics_dt

            # Compute relative acceleration
            x_ddot_rel = (force - self.c * x_dot_rel - self.k * x_rel - self.m * a_ground) / self.m

            # Update relative velocity and displacement
            x_dot_rel += x_ddot_rel * self.dynamics_dt
            x_rel += x_dot_rel * self.dynamics_dt

            # Update state
            self.state = np.array([x_rel, x_dot_rel])

            # Compute absolute acceleration
            x_ddot_mass = a_ground + x_ddot_rel

            # Compute absolute position
            x_mass = self.x_base + x_rel

            if self.enable_plots:
                # Update data buffers for rendering
                self.force_history.append(force)  # Record the force at dynamics_dt intervals
                self.disp_data.append(x_rel)
                self.vel_data.append(x_dot_rel)
                self.mass_accel_data.append(x_ddot_mass)
                self.accel_ground_data.append(a_ground)
                self.force_data.append(force)
                self.time_data.append(self.current_time)

                # Update full-length data buffers for the episode
                self.full_force_history.append(force)
                self.full_disp_data.append(x_rel)
                self.full_vel_data.append(x_dot_rel)
                self.full_mass_accel_data.append(x_ddot_mass)
                self.full_accel_ground_data.append(a_ground)
                self.full_time_data.append(self.current_time)
            else:
                # If plots are disabled, no need to update data buffers
                pass

            # Render intermediate steps
            if self.render_mode == "human":
                self._render_visualization()

        # Calculate reward
        # Reward is negative absolute relative displacement to minimize it
        reward = -abs(self.state[0])

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
        pygame.display.set_caption("Mass-Spring-Damper Simulation with Detailed Plots")
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 18)
        self.clock = pygame.time.Clock()
        # self.last_render_time = pygame.time.get_ticks() / 1000.0  # Initialize last render time

    def _render_visualization(self):
        # Handle events to prevent Pygame from freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Clear screen
        self.screen.fill((255, 255, 255))  # White background

        # Extract state variables
        x_rel, x_dot_rel = self.state

        # Compute positions
        # First rendering (fixed box)
        box1_left = self.first_box_left
        box1_top = self.box_top

        mass1_x = box1_left + self.box_width / 2 + x_rel * self.scale_displacement
        mass1_y = box1_top + self.box_height / 2

        # Second rendering (moving box)
        box2_left = self.second_box_left + self.x_base * self.scale_displacement
        box2_top = self.box_top

        mass2_x = box2_left + self.box_width / 2 + x_rel * self.scale_displacement
        mass2_y = box2_top + self.box_height / 2


        # Add text labels above the local and global reference frame plots
        local_ref_text = self.small_font.render("Local Reference Frame", True, (0, 0, 0))
        global_ref_text = self.small_font.render("Global Reference Frame", True, (0, 0, 0))
    
        # Position the text above the boxes
        self.screen.blit(local_ref_text, (box1_left + (self.box_width // 4), box1_top - 40))  # Local frame text
        self.screen.blit(global_ref_text, (box2_left + (self.box_width // 4), box2_top - 40))  # Global frame text

        # Draw the first box (local reference frame fixed)
        pygame.draw.rect(self.screen, (200, 200, 200), (box1_left, box1_top, self.box_width, self.box_height), 2)

        # Draw mass inside the first box
        pygame.draw.circle(self.screen, (255, 0, 0), (int(mass1_x), int(mass1_y)), self.mass_radius)

        # Draw spring connected to the mass from the left inside wall of the rectangle
        wall1_x = box1_left + 10  # Left wall inside the box
        wall1_y = mass1_y

        # Draw spring as a line in the first box
        spring_color = (0, 255, 0)
        pygame.draw.line(self.screen, spring_color, (wall1_x, wall1_y), (mass1_x - self.mass_radius, mass1_y), 3)

        # Draw damper in the first box
        damper_color = (0, 0, 255)
        damper_width = 10
        damper_height = 20
        damper_x = (wall1_x + mass1_x - self.mass_radius - damper_width) / 2
        damper_y = mass1_y - damper_height / 2
        pygame.draw.rect(self.screen, damper_color, (damper_x, damper_y, damper_width, damper_height))

        # Draw the second box (moving local reference frame)
        pygame.draw.rect(self.screen, (200, 200, 200), (box2_left, box2_top, self.box_width, self.box_height), 2)

        # Draw mass inside the second box
        pygame.draw.circle(self.screen, (255, 0, 0), (int(mass2_x), int(mass2_y)), self.mass_radius)

        # Draw spring connected to the mass from the left inside wall of the rectangle
        wall2_x = box2_left + 10  # Left wall inside the box
        wall2_y = mass2_y

        # Draw spring as a line in the second box
        pygame.draw.line(self.screen, spring_color, (wall2_x, wall2_y), (mass2_x - self.mass_radius, mass2_y), 3)

        # Draw damper in the second box
        damper_x = (wall2_x + mass2_x - self.mass_radius - damper_width) / 2
        damper_y = mass2_y - damper_height / 2
        pygame.draw.rect(self.screen, damper_color, (damper_x, damper_y, damper_width, damper_height))

        # Draw vertical line representing fixed global reference frame
        global_ref_x = self.second_box_left - 50
        pygame.draw.line(self.screen, (0, 0, 0), (global_ref_x, 50), (global_ref_x, self.HEIGHT - 50), 2)

        # Draw force arrow on the mass in the second box
        self._draw_force_arrow(mass2_x, mass2_y)

        # Draw text information
        if self.start_time:
            current_time = (pygame.time.get_ticks() - self.start_time) / 1000  # seconds
        else:
            current_time = 0.0
        info_text = f"Time: {current_time:.2f} s | Rel Displacement: {self.state[0]:.3f} m | Rel Velocity: {self.state[1]:.3f} m/s | Force: {self.last_force:.2f} N"
        text_surface = self.font.render(info_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        if self.enable_plots:
            # Draw plots
            # Draw plots
            # Draw Displacement Plot with dynamic scaling and dynamic y-axis labels
            self._draw_plot(self.disp_data, self.plot_x, self.plot_y_disp, self.plot_width, self.plot_height,
                           (0, 0, 255), "Relative Displacement", "m")

            # Draw Velocity Plot with dynamic scaling and dynamic y-axis labels
            self._draw_plot(self.vel_data, self.plot_x, self.plot_y_vel, self.plot_width, self.plot_height,
                           (0, 200, 0), "Relative Velocity", "m/s")

            # Draw Mass Acceleration Plot with dynamic scaling and dynamic y-axis labels
            self._draw_plot(self.mass_accel_data, self.plot_x, self.plot_y_mass_accel, self.plot_width, self.plot_height,
                           (105, 105, 105), "Mass Acceleration", "m/s²")

            # Draw Ground Acceleration Plot with dynamic scaling and dynamic y-axis labels
            self._draw_plot(self.accel_ground_data, self.plot_x, self.plot_y_accel_ground, self.plot_width, self.plot_height,
                           (255, 0, 0), "Ground Acceleration", "m/s²")

            # Titles for plots
            disp_title = self.small_font.render(f"Relative Displacement vs Time (Last {self.T_max} s)", True, (0, 0, 0))
            self.screen.blit(disp_title, (self.plot_x, self.plot_y_disp + self.plot_height + 5))
            vel_title = self.small_font.render(f"Relative Velocity vs Time (Last {self.T_max} s)", True, (0, 0, 0))
            self.screen.blit(vel_title, (self.plot_x, self.plot_y_vel + self.plot_height + 5))
            mass_accel_title = self.small_font.render(f"Mass Acceleration vs Time (Last {self.T_max} s)", True, (0, 0, 0))
            self.screen.blit(mass_accel_title, (self.plot_x, self.plot_y_mass_accel + self.plot_height + 5))
            accel_ground_title = self.small_font.render(f"Ground Acceleration vs Time (Last {self.T_max} s)", True, (0, 0, 0))
            self.screen.blit(accel_ground_title, (self.plot_x, self.plot_y_accel_ground + self.plot_height + 5))
            
        # Update the display
        pygame.display.flip()
        self.clock.tick(self.fps)

       
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
           observation (np.array): Continuous observation [x_rel, x_dot_rel]
       Returns:
           tuple: Discrete state indices (x_bin, x_dot_bin)
       """
       x, x_dot = observation

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
    # (Rest of the class methods remain unchanged)
