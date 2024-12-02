import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters
m = 1.0                 # Mass (kg)
freq_n = 1.0            # Natural frequency (Hz)
omega_n = freq_n * 2 * math.pi  # Natural angular frequency (rad/s)

k = omega_n ** 2 * m    # Stiffness (N/m)

A = 1.0                 # Amplitude of ground acceleration (m/s²)
freq = 4             # Excitation frequency (Hz)
omega = freq * 2 * math.pi  # Excitation angular frequency (rad/s)

# Time parameters
t_start = 0.0
t_end = 10.0
dt = 0.01              # Time step (s)
t = np.arange(t_start, t_end, dt)

# Initial conditions
x0 = 0.0                # Initial displacement (m)
v0 = 0.0                # Initial velocity (m/s)

# Analytical Solution
denominator = omega_n**2 - omega**2
if denominator != 0:
    # Particular solution
    X_p = (-A) / denominator
    x_p = X_p * np.sin(omega * t)
    x_p0 = X_p * np.sin(omega * 0)
    x_p_prime0 = X_p * omega * np.cos(omega * 0)

    # Homogeneous solution constants
    C1 = x0 - x_p0
    C2 = (v0 - x_p_prime0) / omega_n

    # Total analytical solution
    x_analytical = x_p + C1 * np.cos(omega_n * t) + C2 * np.sin(omega_n * t)
    v_analytical = np.gradient(x_analytical, dt)
else:
    print("Resonance condition: omega equals omega_n. Analytical solution is not defined.")
    x_analytical = None
    v_analytical = None

# Numerical Solution using Euler's method
x_numerical = np.zeros_like(t)
v_numerical = np.zeros_like(t)
x_numerical[0] = x0
v_numerical[0] = v0

for i in range(1, len(t)):
    # Ground acceleration at current time
    a_ground = A * np.sin(omega * t[i-1])
    # Relative acceleration
    a_relative = (-k / m) * x_numerical[i-1] - a_ground
    # Update velocity and displacement
    v_numerical[i] = v_numerical[i-1] + a_relative * dt
    x_numerical[i] = x_numerical[i-1] + v_numerical[i] * dt

# Plotting
plt.figure(figsize=(12, 12))

# Displacement vs Time
plt.subplot(3, 1, 1)
plt.plot(t, x_numerical, label='Numerical Displacement', linestyle='--')
if x_analytical is not None:
    plt.plot(t, x_analytical, label='Analytical Displacement', alpha=0.7)
plt.title('Displacement vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.grid(True)

# Velocity vs Time
plt.subplot(3, 1, 2)
plt.plot(t, v_numerical, label='Numerical Velocity', linestyle='--')
if v_analytical is not None:
    plt.plot(t, v_analytical, label='Analytical Velocity', alpha=0.7)
plt.title('Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

# Phase Plot: Displacement vs Velocity
plt.subplot(3, 1, 3)
plt.plot(x_numerical, v_numerical, label='Numerical Trajectory', linestyle='--')
if x_analytical is not None and v_analytical is not None:
    plt.plot(x_analytical, v_analytical, label='Analytical Trajectory', alpha=0.7)
plt.title('Phase Plot: Velocity vs Displacement')
plt.xlabel('Displacement (m)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
