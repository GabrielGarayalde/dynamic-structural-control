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
    k = 1     # Spring constant
    c = 0.4   # Damping coefficient
    x, x_dot = state
    x_ddot = (-k * x - c * x_dot + u) / m
    return [x_dot, x_ddot]

# Generate training data
dt = 0.01  # Time step
t_train = np.arange(0, 4, dt)
n_states = 2
n_controls = 1

# Generate training data with various control inputs
u_train = np.sin(t_train)  # Sinusoidal input for better system identification

# Initial condition
x0 = [1.0, 0.5]  # Initial displacement and velocity

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

# MPC Parameters
N = 20          # Prediction horizon
Q = np.diag([100, 1])  # State weights
R_values = [0.01, 0.1, 0.2]  # Control weights for parametric study
u_max = 10.0    # Maximum control input
u_min = -10.0   # Minimum control input

# Discretize the system
Ad = np.eye(n_states) + A * dt
Bd = B * dt

# Define the MPC cost function with R as a parameter
def mpc_cost(u_sequence, current_state, reference, R):
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

# Simulation Parameters
t_sim = np.arange(0, 4, dt)
x_ref = np.zeros(n_states)  # Reference state
initial_state = np.array([1.5, 0.8])  # Initial state

# Initialize dictionaries to store simulation results for each R
results = {}
for R in R_values:
    print(f"\nRunning MPC with R = {R}")
    x = initial_state.copy()
    X_sim = [x]
    U_sim = []
    
    # Run MPC
    for t_step in range(len(t_sim) - N):
        # Initialize control sequence
        u_init = np.zeros(N)
        
        # Define bounds for all control inputs in the sequence
        bounds = [(u_min, u_max)] * N
        
        # Optimize control sequence
        result = minimize(
            mpc_cost,
            u_init,
            args=(x, x_ref, R),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-4}
        )
        
        # Get optimal control input
        if result.success:
            u = result.x[0]
        else:
            print(f"Optimization failed at time {t_step*dt:.2f}s for R = {R}")
            u = 0.0
        
        # Apply control input
        U_sim.append(u)
        
        # Simulate system
        x = odeint(mass_spring_damper, x, [0, dt], args=(u,))[-1]
        X_sim.append(x)
    
    # Store results
    results[R] = {
        'X_sim': np.array(X_sim),
        'U_sim': np.array(U_sim)
    }

# Plotting Results
plt.figure(figsize=(15, 10))

# Plot Position
plt.subplot(3,1,1)
for R in R_values:
    plt.plot(t_sim[:len(results[R]['X_sim'])], results[R]['X_sim'][:, 0], label=f'R = {R}')
plt.plot(t_sim[:len(results[R]['X_sim'])], np.zeros_like(t_sim[:len(results[R]['X_sim'])]), 'k--', label='Reference')
plt.ylabel('Position (m)')
plt.title('MPC: Position vs Time for Different R Values')
plt.legend()
plt.grid(True)

# Plot Velocity
plt.subplot(3,1,2)
for R in R_values:
    plt.plot(t_sim[:len(results[R]['X_sim'])], results[R]['X_sim'][:, 1], label=f'R = {R}')
plt.plot(t_sim[:len(results[R]['X_sim'])], np.zeros_like(t_sim[:len(results[R]['X_sim'])]), 'k--', label='Reference')
plt.ylabel('Velocity (m/s)')
plt.title('MPC: Velocity vs Time for Different R Values')
plt.legend()
plt.grid(True)

# Plot Control Input
plt.subplot(3,1,3)
for R in R_values:
    plt.plot(t_sim[:len(results[R]['U_sim'])], results[R]['U_sim'], label=f'R = {R}')
plt.axhline(y=u_max, color='r', linestyle='--', alpha=0.3, label='Control Limits')
plt.axhline(y=u_min, color='r', linestyle='--', alpha=0.3)
plt.ylabel('Control Input (N)')
plt.xlabel('Time (s)')
plt.title('MPC: Control Input vs Time for Different R Values')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Phase Space Plot
plt.figure(figsize=(8,6))
for R in R_values:
    plt.plot(results[R]['X_sim'][:, 0], results[R]['X_sim'][:, 1], label=f'R = {R}')
plt.plot(x_ref[0], x_ref[1], 'k*', markersize=10, label='Reference')
plt.xlabel('Position (m)')
plt.ylabel('Velocity (m/s)')
plt.title('MPC: Phase Space Trajectories for Different R Values')
plt.legend()
plt.grid(True)
plt.show()
