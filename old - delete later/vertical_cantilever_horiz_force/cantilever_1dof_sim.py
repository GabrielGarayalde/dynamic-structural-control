import pygame
import sys
import math
from collections import deque

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1200, 900  # Increased height to accommodate three plots
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("1DOF Vertical Cantilever System Simulation with Pseudo-Force")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 200, 0)
GRAY = (220, 220, 220)
DARK_GRAY = (169, 169, 169)

# Fonts
font = pygame.font.SysFont(None, 24)
small_font = pygame.font.SysFont(None, 20)

# Frames per second
FPS = 30
clock = pygame.time.Clock()

# Fixed system parameters
E = 2.1e9       # Young's Modulus in Pascals (Pa)
I = 1e-6        # Moment of Inertia in m^4
L = 1.5         # Length of the cantilever in meters (m)
m = 10.0        # Mass in kilograms (kg)
x0 = 0.0        # Initial displacement in meters (m)
v0 = 0.0        # Initial velocity in meters per second (m/s)

# Calculate stiffness k for a cantilever beam (first mode)
k = 3 * E * I / (L ** 3)

# Angular frequency
omega = math.sqrt(k / m)

print("System Parameters:")
print(f"Young's Modulus E = {E} Pa")
print(f"Moment of Inertia I = {I} m^4")
print(f"Length L = {L} m")
print(f"Mass m = {m} kg")
print(f"Initial Displacement x0 = {x0} m")
print(f"Initial Velocity v0 = {v0} m/s")
print(f"Calculated stiffness k = {k:.2f} N/m")
print(f"Angular frequency ω = {omega:.2f} rad/s")

# Physics variables
x = x0      # Displacement (horizontal)
v = v0      # Velocity

# Time variables
dt = 1 / FPS  # Time step
elapsed_time = 0.0
T_max = 10.0   # Cutoff time for plotting in seconds

# Scaling factors for visualization
scale_displacement = 150          # pixels per meter for mass displacement
scale_length = 300                 # pixels per meter for vertical beam length
scale_plot_displacement = 100      # pixels per meter for displacement plot
scale_plot_velocity = 30           # pixels per (meter/second) for velocity plot
scale_plot_acceleration = 30       # pixels per (meter/second^2) for acceleration plot

# Ground movement parameters (Reference Frame Movement)
amplitude_ground = 0.1           # Amplitude in meters
frequency_ground = 2             # Frequency in Hz

# Base position for cantilever (fixed)
base_x_initial = WIDTH // 4
base_y = 3 * HEIGHT // 4  # Positioned higher to accommodate plots below

# Mass properties
mass_radius = 10

# Beam discretization
N_segments = 20  # Number of segments to discretize the beam
z_values = [i * L / N_segments for i in range(N_segments + 1)]  # Positions along the beam

# Plot properties
plot_width = WIDTH - base_x_initial - 300
plot_height = HEIGHT // 4 - 100  # Adjusted for three plots
plot_x = base_x_initial + 200
plot_y_disp = HEIGHT // 4
plot_y_vel = plot_y_disp + plot_height + 50
plot_y_accel = plot_y_vel + plot_height + 50

# Data buffers for plotting
max_points = int(T_max * FPS)
disp_data = deque(maxlen=max_points)
vel_data = deque(maxlen=max_points)
accel_data = deque(maxlen=max_points)
time_data = deque(maxlen=max_points)

# Initialize data buffers with initial conditions
for _ in range(max_points):
    disp_data.append(x0)    # Set initial displacement to x0
    vel_data.append(v0)     # Set initial velocity to v0
    accel_data.append(0.0)  # Initial acceleration is zero
    time_data.append(-T_max + _ * dt)

# Helper function to draw the beam based on discretized points
def draw_beam(screen, base_x, base_y, x_displacement, z_values, N, scale_deflection, scale_length, scale_displacement, color=BLACK, width=4):
    beam_points = []
    for z in z_values:
        # Calculate deflection based on mode shape (quadratic)
        # y(z, t) = x(t) * (z/L)^2
        phi = (z / L) ** 2
        x_deflection = x_displacement * phi
        # Map to screen coordinates
        beam_x = base_x + x_deflection * scale_displacement  # Horizontal displacement
        beam_y = base_y - z * scale_length  # Vertical position
        beam_points.append((beam_x, beam_y))
    pygame.draw.lines(screen, color, False, beam_points, width)

# Function to draw individual plots
def draw_plot(data, plot_x, plot_y, plot_width, plot_height, scale, color, label, y_label):
    pygame.draw.rect(screen, GRAY, (plot_x, plot_y, plot_width, plot_height))
    pygame.draw.rect(screen, BLACK, (plot_x, plot_y, plot_width, plot_height), 2)
    # Axes
    pygame.draw.line(screen, BLACK, (plot_x, plot_y + plot_height//2),
                     (plot_x + plot_width, plot_y + plot_height//2), 1)  # X-axis
    pygame.draw.line(screen, BLACK, (plot_x, plot_y),
                     (plot_x, plot_y + plot_height), 1)  # Y-axis

    # Plot data
    if len(data) > 1:
        points = []
        for i in range(len(data)):
            t = time_data[i]
            value = data[i]
            # Only plot data within the cutoff time window
            if t < elapsed_time - T_max:
                continue
            # Map time to x-coordinate
            x_coord = plot_x + (t - (elapsed_time - T_max)) / T_max * plot_width
            # Map value to y-coordinate
            y_coord = plot_y + plot_height//2 - value * scale
            points.append((x_coord, y_coord))
        if points:
            pygame.draw.lines(screen, color, False, points, 2)

    # Labels
    label_surface = font.render(label, True, BLACK)
    screen.blit(label_surface, (plot_x, plot_y - 30))
    # Y-axis labels
    pygame.draw.line(screen, BLACK, (plot_x - 5, plot_y + plot_height//2 - 1),
                     (plot_x + 5, plot_y + plot_height//2 -1), 1)  # Center
    pygame.draw.line(screen, BLACK, (plot_x - 5, plot_y + plot_height//2 - scale),
                     (plot_x + 5, plot_y + plot_height//2 - scale), 1)  # Positive
    y_label_surface = font.render(f"{1} {y_label}", True, BLACK)
    screen.blit(y_label_surface, (plot_x + 10, plot_y + plot_height//2 - scale - 10))

# Main loop
running = True
while running:
    clock.tick(FPS)
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update elapsed time
    elapsed_time += dt

    # Calculate ground displacement and acceleration (pseudo-force)
    x_ground = amplitude_ground * math.sin(2 * math.pi * frequency_ground * elapsed_time)  # Ground displacement
    a_ground = - (2 * math.pi * frequency_ground) ** 2 * amplitude_ground * math.sin(2 * math.pi * frequency_ground * elapsed_time)  # Ground acceleration

    # Physics integration using Euler method with pseudo-force
    # x'' = - (k/m) x + a_ground
    a = - (k / m) * x + a_ground
    v += a * dt
    x += v * dt

    # Update data buffers
    disp_data.append(x)
    vel_data.append(v)
    accel_data.append(a_ground)  # Store pseudo-force (acceleration)
    time_data.append(elapsed_time)

    # Clear screen
    screen.fill(WHITE)

    # Draw fixed base
    pygame.draw.circle(screen, BLACK, (base_x_initial, base_y), 10)

    # Calculate mass position (fixed vertically with deflection)
    mass_x = base_x_initial + x * scale_displacement
    mass_y = base_y - L * scale_length  # Mass is at the end of the beam vertically

    # Draw the beam based on displacement
    draw_beam(screen, base_x_initial, base_y, x, z_values, N_segments, scale_deflection=1, scale_length=scale_length, scale_displacement=scale_displacement, color=BLACK, width=4)

    # Draw mass as a circle
    pygame.draw.circle(screen, RED, (int(mass_x), int(mass_y)), mass_radius)

    # Draw text information
    info_text = f"Time: {elapsed_time:.2f} s | Displacement: {x:.3f} m | Velocity: {v:.3f} m/s | Ground Acceleration: {a_ground:.3f} m/s²"
    text_surface = font.render(info_text, True, BLACK)
    screen.blit(text_surface, (10, 10))

    # Draw Displacement Plot
    draw_plot(disp_data, plot_x, plot_y_disp, plot_width, plot_height,
              scale_plot_displacement, BLUE, "Displacement (m)", "m")

    # Draw Velocity Plot
    draw_plot(vel_data, plot_x, plot_y_vel, plot_width, plot_height,
              scale_plot_velocity, GREEN, "Velocity (m/s)", "m/s")

    # Draw Acceleration Plot
    draw_plot(accel_data, plot_x, plot_y_accel, plot_width, plot_height,
              scale_plot_acceleration, DARK_GRAY, "Acceleration (m/s²)", "m/s²")

    # Titles for plots
    disp_title = small_font.render(f"Displacement vs Time (Last {T_max} s)", True, BLACK)
    screen.blit(disp_title, (plot_x, plot_y_disp + plot_height + 5))
    vel_title = small_font.render(f"Velocity vs Time (Last {T_max} s)", True, BLACK)
    screen.blit(vel_title, (plot_x, plot_y_vel + plot_height + 5))
    accel_title = small_font.render(f"Ground Acceleration vs Time (Last {T_max} s)", True, BLACK)
    screen.blit(accel_title, (plot_x, plot_y_accel + plot_height + 5))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
