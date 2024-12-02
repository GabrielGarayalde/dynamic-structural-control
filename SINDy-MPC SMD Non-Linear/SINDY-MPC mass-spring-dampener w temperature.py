import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from pysindy import SINDy
from pysindy.optimizers import STLSQ
from pysindy.feature_library import PolynomialLibrary

# Define the system dynamics (mass-spring-damper with temperature-dependent damping)
def mass_spring_damper(state, t, u, T):
    m = 1.0   # Mass
    k = 1.0   # Spring constant
    c = 0.4 + 0.1 * T   # Damping coefficient depends on temperature
    x, x_dot = state
    x_ddot = (-k * x - c * x_dot + u) / m
    return [x_dot, x_ddot]

# Generate training data
dt = 0.01  # Time step
t_train = np.arange(0, 10, dt)
n_states = 2
n_controls = 1

# Generate training data with various control inputs
u_train = np.sin(t_train)  # Sinusoidal input for better system identification

# Generate temperature data
T0 = 25.0       # Mean temperature
A_T = 5.0       # Amplitude of temperature variation
omega_T = 0.5   # Frequency of temperature variation
T_train = T0 + A_T * np.sin(omega_T * t_train)

# Initial condition
x0 = [0, 0.5]  # Initial displacement and velocity

# Simulate the system to generate training data
X_train = np.zeros((len(t_train), 2))
X_train[0] = x0
for i in range(len(t_train)-1):
    sol = odeint(
        mass_spring_damper, 
        X_train[i], 
        [t_train[i], t_train[i+1]], 
        args=(u_train[i], T_train[i])
    )
    X_train[i+1] = sol[-1]

# Prepare data for SINDy
X_with_control = np.column_stack((X_train, u_train, T_train))

# Calculate derivatives for SINDy
X_dot = np.array([
    mass_spring_damper(state, t, u, T) 
    for state, t, u, T in zip(X_train, t_train, u_train, T_train)
])

# Fit SINDy model
poly_library = PolynomialLibrary(degree=2, include_bias=False)
optimizer = STLSQ(alpha=0.001, threshold=0.05)
model = SINDy(feature_library=poly_library, optimizer=optimizer)
model.fit(X_with_control, x_dot=X_dot, t=dt)
model.print()

# Extract system matrices (including temperature as input)
coefficients = model.coefficients()
n_inputs = 2  # Control input and temperature
A = coefficients[:, :n_states]
B = coefficients[:, n_states:n_states + n_controls]   # Control input coefficients
D = coefficients[:, -1].reshape(-1, 1)                # Temperature coefficients

print(f"A shape: {A.shape}, B shape: {B.shape}, D shape: {D.shape}")

# MPC parameters
N = 20          # Prediction horizon
Q = np.diag([100, 1])  # State weights
R = 0.01        # Control weight
u_max = 10.0    # Maximum control input
u_min = -10.0   # Minimum control input

# Discretize the system
Ad = np.eye(n_states) + A * dt
Bd = B * dt
Dd = D * dt

# Simulation setup
t_sim = np.arange(0, 4, dt)
x_ref = np.zeros(n_states)  # Reference state
x0_sim = np.array([2.0, 0.8])    # Initial state

# Generate temperature data for simulation
T_sim = T0 + A_T * np.sin(omega_T * t_sim)

# Initialize arrays for controlled simulation
X_sim = [x0_sim]
U_sim = []

# Run MPC
for idx in range(len(t_sim) - N):
    # Current time and temperature
    current_time = t_sim[idx]
    current_temperature = T_sim[idx]
    x = X_sim[-1]
    
    # Initialize control sequence
    u_init = np.zeros(N)
    
    # Define bounds for all control inputs in the sequence
    bounds = [(u_min, u_max)] * N
    
    # Define the MPC cost function
    def mpc_cost(u_sequence, current_state, reference):
        cost = 0
        x = current_state.copy()
        idx_inner = idx  # Start index for temperature
        
        for i in range(len(u_sequence)):
            # Clip control input to bounds
            u = np.clip(u_sequence[i], u_min, u_max)
            
            # Get temperature at future time step
            if idx_inner + i < len(T_sim):
                T_future = T_sim[idx_inner + i]
            else:
                T_future = T_sim[-1]  # Use last temperature if out of bounds
            
            # State cost
            error = x - reference
            cost += error.T @ Q @ error
            
            # Control cost
            cost += R * u**2
            
            # Propagate state
            x = Ad @ x + Bd.flatten() * u + Dd.flatten() * T_future
        
        return cost
    
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
        print(f"Optimization failed at time {current_time:.2f}")
        u = 0.0
    
    # Apply control input
    U_sim.append(u)
    
    # Simulate system with control input
    x_next = odeint(
        mass_spring_damper, 
        x, 
        [t_sim[idx], t_sim[idx+1]], 
        args=(u, current_temperature)
    )[-1]
    X_sim.append(x_next)

# Convert results to arrays
X_sim = np.array(X_sim)
U_sim = np.array(U_sim)

# Simulate free vibration response (no control input)
X_free = [x0_sim]
for idx in range(len(t_sim)-1):
    current_temperature = T_sim[idx]
    u_free = 0.0  # No control input
    x_free_next = odeint(
        mass_spring_damper, 
        X_free[-1], 
        [t_sim[idx], t_sim[idx+1]], 
        args=(u_free, current_temperature)
    )[-1]
    X_free.append(x_free_next)

X_free = np.array(X_free)

# Plot results
plt.figure(figsize=(12, 8))

# Position plot
plt.subplot(3,1,1)
plt.plot(t_sim[:len(X_sim)], X_sim[:, 0], label='Position (Controlled)')
plt.plot(t_sim[:len(X_free)], X_free[:, 0], label='Position (Free Vibration)')
plt.plot(t_sim[:len(X_sim)], np.zeros_like(t_sim[:len(X_sim)]), 'r--', label='Reference')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)

# Velocity plot
plt.subplot(3,1,2)
plt.plot(t_sim[:len(X_sim)], X_sim[:, 1], label='Velocity (Controlled)')
plt.plot(t_sim[:len(X_free)], X_free[:, 1], label='Velocity (Free Vibration)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

# Control input plot
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
