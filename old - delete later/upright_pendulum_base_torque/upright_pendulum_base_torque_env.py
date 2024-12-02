# upright_pendulum_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class UprightPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.max_speed = 8
        self.max_torque = 1.0
        self.dynamics_dt = 0.05  # Dynamics time step
        self.action_dt = 0.2     # Action time step
        self.num_integration_steps = int(self.action_dt / self.dynamics_dt)        
        self.g = 10.0  # gravity
        self.m = 1.0   # mass
        self.l = 1.0   # length

        # Define action and observation space
        # Continuous action space will be discretized later
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
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

        # Initialize last torque
        self.last_torque = 0.0

        # Font for displaying text
        self.font = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start upright (theta = 0) with a small random noise
        high = np.array([0.05, 0.05])  # Small perturbation
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_torque = 0.0  # Reset last torque
        self.torque_history = []  # Initialize torque history
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}


    def _get_obs(self):
        theta, theta_dot = self.state
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def step(self, action):
        torque = np.clip(action, -self.max_torque, self.max_torque)[0]
        self.last_torque = torque
        self.torque_history.append(torque)
    
        for _ in range(self.num_integration_steps):
            theta, theta_dot = self.state
    
            # Dynamics equations using the smaller dynamics_dt
            new_theta_dot = theta_dot + (
                -3 * self.g / (2 * self.l) * np.sin(theta)
                + 3.0 / (self.m * self.l ** 2) * torque
            ) * self.dynamics_dt
            new_theta_dot = np.clip(new_theta_dot, -self.max_speed, self.max_speed)
            new_theta = theta + new_theta_dot * self.dynamics_dt
    
            self.state = np.array([new_theta, new_theta_dot])
    
            # Optional: Render intermediate steps
            if self.render_mode == "human":
                self.render()
    
        # After integration, render the final state
        if self.render_mode == "human":
            self.render()
    
        # Calculate reward based on the final state
        theta, theta_dot = self.state
        uprightness = np.cos(theta)
        reward = uprightness
    
        terminated = False
        truncated = False
    
        return self._get_obs(), reward, terminated, truncated, {}


    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((500, 500))
            pygame.display.set_caption("Upright Pendulum")
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
    
        # Draw the curved torque arrow at the base
        torque = self.last_torque
        max_torque = self.max_torque
    
        # Set arc parameters
        arc_radius = 30
        arc_center = origin
        arc_rect = pygame.Rect(
            arc_center[0] - arc_radius,
            arc_center[1] - arc_radius,
            2 * arc_radius,
            2 * arc_radius
        )
    
        # Determine start and end angles based on torque direction and magnitude
        if torque != 0:
            # Calculate the angular span proportional to torque magnitude
            angular_span = (abs(torque) / max_torque) * 2 * np.pi  # Up to 2π radians
    
            if torque > 0:
                # Positive torque: anti-clockwise arc from 0 to angular_span
                start_angle = -np.pi / 2  # Starting from upwards (-90 degrees)
                end_angle = start_angle + angular_span
                arc_color = (255, 0, 0)  # Red color for torque arrow
            else:
                # Negative torque: clockwise arc from 0 to -angular_span
                              
                start_angle = -np.pi / 2 - angular_span # Starting from upwards (-90 degrees)
                end_angle = -np.pi / 2
                arc_color = (0, 255, 0)  # Red color for torque arrow

            # Thickness of the arc remains constant
            thickness = 5
    
            # Draw the arc
            pygame.draw.arc(self.screen, arc_color, arc_rect, start_angle, end_angle, thickness)
    
            # # Draw the arrowhead at the end of the arc
            # arrowhead_length = 15
            
                
            # arrow_angle = end_angle if torque > 0 else start_angle
    
            # arrow_tip = (
            #     arc_center[0] + arc_radius * np.cos(arrow_angle),
            #     arc_center[1] + arc_radius * np.sin(arrow_angle)
            # )
            # left_wing_angle = arrow_angle + (np.pi / 6) if torque > 0 else arrow_angle - (np.pi / 6)
            # right_wing_angle = arrow_angle - (np.pi / 6) if torque > 0 else arrow_angle + (np.pi / 6)
    
            # left_wing = (
            #     arrow_tip[0] + arrowhead_length * np.cos(left_wing_angle),
            #     arrow_tip[1] + arrowhead_length * np.sin(left_wing_angle)
            # )
            # right_wing = (
            #     arrow_tip[0] + arrowhead_length * np.cos(right_wing_angle),
            #     arrow_tip[1] + arrowhead_length * np.sin(right_wing_angle)
            # )
    
            # pygame.draw.polygon(self.screen, arc_color, [arrow_tip, left_wing, right_wing])
    
        # Display torque magnitude as text
        torque_text = f"Torque: {torque:.2f}"
        text_surface = self.font.render(torque_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))
    
        # Display elapsed time
        elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000  # Convert to seconds
        time_text = f"Time: {elapsed_time:.2f}s"
        time_surface = self.font.render(time_text, True, (0, 0, 0))
        self.screen.blit(time_surface, (400, 10))  # Position at top right
    
        # Update the display
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        self.clock.tick(20)


    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

