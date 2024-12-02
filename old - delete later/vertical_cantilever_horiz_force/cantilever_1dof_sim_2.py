import pygame
import sys
import math
from collections import deque
import random
# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1200, 1200  # Increased height to accommodate four plots
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
m = 100.0        # Mass in kilograms (kg)
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
scale_length = 300                # pixels per meter for vertical beam length

# Ground movement parameters (Reference Frame Movement)
amplitude_ground = 0.0           # Amplitude in meters
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
plot_height = (HEIGHT - 200) / 4 - 50  # Adjusted for four plots
plot_x = base_x_initial + 200
plot_y_disp = 100
plot_y_vel = plot_y_disp + plot_height + 50
plot_y_mass_accel = plot_y_vel + plot_height + 50
plot_y_accel_ground = plot_y_mass_accel + plot_height + 50

# Data buffers for plotting
max_points = int(T_max * FPS)
disp_data = deque(maxlen=max_points)
vel_data = deque(maxlen=max_points)
mass_accel_data = deque(maxlen=max_points)  # New buffer for mass acceleration
accel_ground_data = deque(maxlen=max_points)  # Renamed for clarity
time_data = deque(maxlen=max_points)

# Initialize data buffers with initial conditions
for _ in range(max_points):
    disp_data.append(x0)            # Set initial displacement to x0
    vel_data.append(v0)             # Set initial velocity to v0
    mass_accel_data.append(0.0)     # Initial mass acceleration is zero
    accel_ground_data.append(0.0)   # Initial ground acceleration is zero
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

# Function to calculate a "nice" scale for the y-axis
def get_nice_scale(max_val):
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

# Function to draw individual plots with dynamic scaling and dynamic y-axis labels
def draw_plot(data, plot_x, plot_y, plot_width, plot_height, color, label, y_label):
    pygame.draw.rect(screen, GRAY, (plot_x, plot_y, plot_width, plot_height))
    pygame.draw.rect(screen, BLACK, (plot_x, plot_y, plot_width, plot_height), 2)
    # Axes
    pygame.draw.line(screen, BLACK, (plot_x, plot_y + plot_height//2),
                     (plot_x + plot_width, plot_y + plot_height//2), 1)  # X-axis
    pygame.draw.line(screen, BLACK, (plot_x, plot_y),
                     (plot_x, plot_y + plot_height), 1)  # Y-axis

    # Determine the maximum absolute value in the data for scaling
    max_val = max(abs(val) for val in data) if data else 1.0
    nice_scale = get_nice_scale(max_val)

    # Calculate scaling factor based on nice_scale
    padding = 10  # pixels
    scale = (plot_height / 2 - padding) / nice_scale if nice_scale != 0 else 1

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
    # Center label (0)
    pygame.draw.line(screen, BLACK, (plot_x - 5, plot_y + plot_height//2 - 1),
                     (plot_x + 5, plot_y + plot_height//2 -1), 1)  # Center
    zero_label = font.render("0", True, BLACK)
    screen.blit(zero_label, (plot_x - 20, plot_y + plot_height//2 - 10))
    
    # Positive label
    pygame.draw.line(screen, BLACK, (plot_x - 5, plot_y + plot_height//2 - nice_scale * scale),
                     (plot_x + 5, plot_y + plot_height//2 - nice_scale * scale), 1)  # Positive
    pos_label = font.render(f"{nice_scale} {y_label}", True, BLACK)
    screen.blit(pos_label, (plot_x - 50, plot_y + plot_height//2 - nice_scale * scale - 10))
    
    # Negative label
    pygame.draw.line(screen, BLACK, (plot_x - 5, plot_y + plot_height//2 + nice_scale * scale),
                     (plot_x + 5, plot_y + plot_height//2 + nice_scale * scale), 1)  # Negative
    neg_label = font.render(f"-{nice_scale} {y_label}", True, BLACK)
    screen.blit(neg_label, (plot_x - 60, plot_y + plot_height//2 + nice_scale * scale - 10))

    # Optional: Draw grid lines for better readability
    # Positive grid line
    pygame.draw.line(screen, DARK_GRAY, (plot_x, plot_y + plot_height//2 - nice_scale * scale),
                     (plot_x + plot_width, plot_y + plot_height//2 - nice_scale * scale), 1)
    # Negative grid line
    pygame.draw.line(screen, DARK_GRAY, (plot_x, plot_y + plot_height//2 + nice_scale * scale),
                     (plot_x + plot_width, plot_y + plot_height//2 + nice_scale * scale), 1)

    return scale  # Return scale for potential future use

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
    # x_ground = amplitude_ground * math.sin(2 * math.pi * frequency_ground * elapsed_time)  # Ground displacement
    # a_ground = - (2 * math.pi * frequency_ground) ** 2 * amplitude_ground * math.sin(2 * math.pi * frequency_ground * elapsed_time)  # Ground acceleration

    a_ground = random.uniform(-1, 1) * 10  # White noise in the range [-2, 2] m/s²

    # Physics integration using Euler method with pseudo-force
    # x'' = - (k/m) x + a_ground
    a_mass = - (k / m) * x + a_ground
    v += a_mass * dt
    x += v * dt

    # Update data buffers
    disp_data.append(x)
    vel_data.append(v)
    mass_accel_data.append(a_mass)    # Append mass acceleration
    accel_ground_data.append(a_ground)  # Append ground acceleration
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
    info_text = f"Time: {elapsed_time:.2f} s | Displacement: {x:.3f} m | Velocity: {v:.3f} m/s | Mass Acceleration: {a_mass:.3f} m/s² | Ground Acceleration: {a_ground:.3f} m/s²"
    text_surface = font.render(info_text, True, BLACK)
    screen.blit(text_surface, (10, 10))

    # Draw Displacement Plot with dynamic scaling and dynamic y-axis labels
    draw_plot(disp_data, plot_x, plot_y_disp, plot_width, plot_height,
              BLUE, "Displacement", "m")

    # Draw Velocity Plot with dynamic scaling and dynamic y-axis labels
    draw_plot(vel_data, plot_x, plot_y_vel, plot_width, plot_height,
              GREEN, "Velocity", "m/s")

    # Draw Mass Acceleration Plot with dynamic scaling and dynamic y-axis labels
    draw_plot(mass_accel_data, plot_x, plot_y_mass_accel, plot_width, plot_height,
              DARK_GRAY, "Mass Acceleration", "m/s²")

    # Draw Ground Acceleration Plot with dynamic scaling and dynamic y-axis labels
    draw_plot(accel_ground_data, plot_x, plot_y_accel_ground, plot_width, plot_height,
              RED, "Ground Acceleration", "m/s²")

    # Titles for plots
    disp_title = small_font.render(f"Displacement vs Time (Last {T_max} s)", True, BLACK)
    screen.blit(disp_title, (plot_x, plot_y_disp + plot_height + 5))
    vel_title = small_font.render(f"Velocity vs Time (Last {T_max} s)", True, BLACK)
    screen.blit(vel_title, (plot_x, plot_y_vel + plot_height + 5))
    mass_accel_title = small_font.render(f"Mass Acceleration vs Time (Last {T_max} s)", True, BLACK)
    screen.blit(mass_accel_title, (plot_x, plot_y_mass_accel + plot_height + 5))
    accel_ground_title = small_font.render(f"Ground Acceleration vs Time (Last {T_max} s)", True, BLACK)
    screen.blit(accel_ground_title, (plot_x, plot_y_accel_ground + plot_height + 5))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
