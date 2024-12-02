import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from pysindy import SINDy
from pysindy.optimizers import STLSQ
from pysindy.feature_library import PolynomialLibrary

# Define the system dynamics (mass-spring-damper)
def mass_spring_damper(state, t, u):
    m = 1.0   # Mass
    k = 1   # Spring constant
    c = 0.4   # Damping coefficient
    x, x_dot = state
    x_ddot = (-k * x - c * x_dot + u) / m
    return [x_dot, x_ddot]

# Generate training data
dt = 0.01  # Time step
t_train = np.arange(0, 3, dt)
n_states = 2
n_controls = 1

# Generate training data with various control inputs
u_train = np.sin(t_train)  # Sinusoidal input for better system identification

# Initial condition
x0 = [0, 0.5]  # Initial displacement and velocity

# Simulate the system to generate training data
X_train = np.zeros((len(t_train), 2))
X_train[0] = x0
for i in range(len(t_train)-1):
    sol = odeint(mass_spring_damper, X_train[i], [0, dt], args=(u_train[i],))
    X_train[i+1] = sol[-1]

# Add control input to training data
X_with_control = np.column_stack((X_train, u_train))

# Calculate derivatives for SINDy
X_dot = np.array([mass_spring_damper(state, t, u) for state, t, u in zip(X_train, t_train, u_train)])

# Fit SINDy model
poly_library = PolynomialLibrary(degree=1, include_bias=False)
optimizer = STLSQ(alpha=0.001, threshold=0.05)
model = SINDy(feature_library=poly_library, optimizer=optimizer)
model.fit(X_with_control, x_dot=X_dot, t=dt)
model.print()

# Extract system matrices
coefficients = model.coefficients()
A = coefficients[:, :n_states]
B = coefficients[:, -1].reshape(-1, 1)

print(f"A shape: {A.shape}, B shape: {B.shape}")

# MPC parameters
N = 20          # Prediction horizon
Q = np.diag([100, 1])  # State weights
R = 0.01        # Control weight
u_max = 10.0    # Maximum control input
u_min = -10.0   # Minimum control input

# Discretize the system
Ad = np.eye(n_states) + A * dt
Bd = B * dt

# Define the MPC cost function
def mpc_cost(u_sequence, current_state, reference):
    cost = 0
    x = current_state.copy()
    
    for i in range(len(u_sequence)):
        # Clip control input to bounds
        u = np.clip(u_sequence[i], u_min, u_max)
        
        # State cost
        error = x - reference
        cost += error.T @ Q @ error
        
        # Control cost
        cost += R * u**2
        
        # Propagate state
        x = Ad @ x + Bd.flatten() * u
    
    return cost

# Simulation
t_sim = np.arange(0, 3, dt)
x_ref = np.zeros(n_states)  # Reference state
x = np.array([2.0, 0.8])    # Initial state

X_sim = [x]
U_sim = []

# Run MPC
for t in range(len(t_sim) - N):
    # Initialize control sequence
    u_init = np.zeros(N)
    
    # Define bounds for all control inputs in the sequence
    bounds = [(u_min, u_max)] * N
    
    # Optimize control sequence
    result = minimize(
        mpc_cost,
        u_init,
        args=(x, x_ref),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 100}
    )
    
    # Get optimal control input
    if result.success:
        u = result.x[0]
    else:
        print(f"Optimization failed at time {t*dt:.2f}")
        u = 0.0
    
    # Apply control input
    U_sim.append(u)
    
    # Simulate system
    x = odeint(mass_spring_damper, x, [0, dt], args=(u,))[-1]
    X_sim.append(x)

# Convert results to arrays
X_sim = np.array(X_sim)
U_sim = np.array(U_sim)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(3,1,1)
plt.plot(t_sim[:len(X_sim)], X_sim[:, 0], label='Position')
plt.plot(t_sim[:len(X_sim)], np.zeros_like(t_sim[:len(X_sim)]), 'r--', label='Reference')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t_sim[:len(X_sim)], X_sim[:, 1], label='Velocity')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(t_sim[:len(U_sim)], U_sim, label='Control Input')
plt.ylabel('Control Input (N)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)

# Add control limits to plot
plt.axhline(y=u_max, color='r', linestyle='--', alpha=0.3, label='Control Limits')
plt.axhline(y=u_min, color='r', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

# New Phase Space Plot
plt.figure(figsize=(6,6))
plt.plot(X_sim[:, 0], X_sim[:, 1], label='Trajectory')
plt.plot(x_ref, np.zeros_like(x_ref), 'r--', label='Reference')
plt.xlabel('Position (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Phase Space')
plt.legend()
plt.grid(True)
plt.show()