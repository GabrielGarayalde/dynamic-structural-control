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
N = 20                     # Prediction horizon
Q_configs = [
    np.diag([100, 1]),    # Q = diag([100, 1])
    np.diag([50, 2]),    # Q = diag([10, 10])
    np.diag([20, 5])     # Q = diag([1, 100])
]
Q_labels = ['Q = diag([100, 1])', 'Q = diag([10, 10])', 'Q = diag([1, 100])']
u_max = 10.0               # Maximum control input
u_min = -10.0              # Minimum control input

# Discretize the system
Ad = np.eye(n_states) + A * dt
Bd = B * dt

# Define the MPC cost function with Q as a parameter
def mpc_cost(u_sequence, current_state, reference, Q):
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
x_ref = np.zeros(n_states)          # Reference state
initial_state = np.array([1.5, 0.8])  # Initial state

# Control Weight (Fixed for this study)
R = 0.01  # You can adjust this if needed

# Initialize dictionaries to store simulation results for each Q configuration
results_Q = {}
for Q, label in zip(Q_configs, Q_labels):
    print(f"\nRunning MPC with {label}")
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
            args=(x, x_ref, Q),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-4}
        )
        
        # Get optimal control input
        if result.success:
            u = result.x[0]
        else:
            print(f"Optimization failed at time {t_step*dt:.2f}s for {label}")
            u = 0.0
        
        # Apply control input
        U_sim.append(u)
        
        # Simulate system
        x = odeint(mass_spring_damper, x, [0, dt], args=(u,))[-1]
        X_sim.append(x)
    
    # Store results
    results_Q[label] = {
        'X_sim': np.array(X_sim),
        'U_sim': np.array(U_sim)
    }

# Plotting Results
plt.figure(figsize=(18, 15))

# Plot Position
plt.subplot(4,1,1)
for label in Q_labels:
    plt.plot(t_sim[:len(results_Q[label]['X_sim'])], results_Q[label]['X_sim'][:, 0], label=label)
plt.plot(t_sim[:len(results_Q[label]['X_sim'])], np.zeros_like(t_sim[:len(results_Q[label]['X_sim'])]), 'k--', label='Reference')
plt.ylabel('Position (m)')
plt.title('MPC: Position vs Time for Different Q Configurations')
plt.legend()
plt.grid(True)

# Plot Velocity
plt.subplot(4,1,2)
for label in Q_labels:
    plt.plot(t_sim[:len(results_Q[label]['X_sim'])], results_Q[label]['X_sim'][:, 1], label=label)
plt.plot(t_sim[:len(results_Q[label]['X_sim'])], np.zeros_like(t_sim[:len(results_Q[label]['X_sim'])]), 'k--', label='Reference')
plt.ylabel('Velocity (m/s)')
plt.title('MPC: Velocity vs Time for Different Q Configurations')
plt.legend()
plt.grid(True)

# Plot Control Input
plt.subplot(4,1,3)
for label in Q_labels:
    plt.plot(t_sim[:len(results_Q[label]['U_sim'])], results_Q[label]['U_sim'], label=label)
plt.axhline(y=u_max, color='r', linestyle='--', alpha=0.3, label='Control Limits')
plt.axhline(y=u_min, color='r', linestyle='--', alpha=0.3)
plt.ylabel('Control Input (N)')
plt.title('MPC: Control Input vs Time for Different Q Configurations')
plt.legend()
plt.grid(True)

# Adjust layout before showing
plt.tight_layout()

# Show state and control plots
plt.show()

# Phase Space Plot (Separate)
plt.figure(figsize=(8,6))
for label in Q_labels:
    plt.plot(results_Q[label]['X_sim'][:, 0], results_Q[label]['X_sim'][:, 1], label=label)
plt.plot(x_ref[0], x_ref[1], 'k*', markersize=10, label='Reference')
plt.xlabel('Position (m)')
plt.ylabel('Velocity (m/s)')
plt.title('MPC: Phase Space Trajectories for Different Q Configurations')
plt.legend()
plt.grid(True)
plt.show()
