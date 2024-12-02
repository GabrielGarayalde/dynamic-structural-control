import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pysindy import SINDy
from pysindy.optimizers import STLSQ
from pysindy.feature_library import PolynomialLibrary
from scipy.optimize import minimize

# --------------------------- Modify System Dynamics ---------------------------
def mass_spring_damper_nonlinear(state, t, u):
    """
    Nonlinear mass-spring-damper system with a cubic spring term.

    Parameters:
    - state: [x, x_dot], where x is position and x_dot is velocity.
    - t: Time variable (not used as dynamics are autonomous except for control).
    - u: Control input.

    Returns:
    - [x_dot, x_ddot]: Derivatives of position and velocity.
    """
    m = 1.0    # Mass
    k = 1.0    # Linear spring constant
    c = 0.4    # Damping coefficient
    k_nl = 0.1 # Nonlinear (cubic) spring coefficient

    x, x_dot = state
    x_ddot = (-k * x - c * x_dot - k_nl * x**3 + u) / m
    return [x_dot, x_ddot]

# --------------------------- Generate Training Data ---------------------------
dt = 0.01  # Time step
t_train = np.arange(0, 3, dt)  # Extended time for better identification
n_states = 2
n_controls = 1

# Generate control input (sinusoidal input)
u_train = np.sin(2 * t_train)  # Increased frequency for richer dynamics

# Initial condition
x0 = [1.0, 0.0]  # Initial displacement and velocity

# Simulate the system to generate training data
X_train = np.zeros((len(t_train), n_states))
X_train[0] = x0
for i in range(len(t_train)-1):
    sol = odeint(
        mass_spring_damper_nonlinear, 
        X_train[i], 
        [t_train[i], t_train[i+1]], 
        args=(u_train[i],)
    )
    X_train[i+1] = sol[-1]

# Prepare data for SINDy
X_with_control = np.column_stack((X_train, u_train))

# Calculate derivatives for SINDy
X_dot = np.array([
    mass_spring_damper_nonlinear(state, t, u) 
    for state, t, u in zip(X_train, t_train, u_train)
])

# --------------------------- Fit SINDy Model ---------------------------
# Use a polynomial library of degree 3 to capture cubic terms
poly_library = PolynomialLibrary(degree=3, include_bias=False)
optimizer = STLSQ(alpha=0.001, threshold=0.05)

model = SINDy(feature_library=poly_library, optimizer=optimizer)
model.fit(X_with_control, x_dot=X_dot, t=dt)
model.print()

# Extract system matrices (coefficients)
coefficients = model.coefficients()
n_features = coefficients.shape[1] - n_controls  # Exclude control input
A = coefficients[:, :n_states]
B = coefficients[:, -n_controls:].reshape(-1, n_controls)

print(f"A shape: {A.shape}, B shape: {B.shape}")

# --------------------------- MPC Parameters ---------------------------
N = 20          # Prediction horizon (increased for nonlinear system)
Q = np.diag([100, 1])  # State weights
R = 0.1         # Control weight
u_max = 10.0    # Maximum control input
u_min = -10.0   # Minimum control input

# Discretize the system (Euler discretization)
Ad = np.eye(n_states) + A * dt
Bd = B * dt

# --------------------------- Simulation Setup ---------------------------
t_sim = np.arange(0, 3, dt)
x_ref = np.zeros(n_states)  # Reference state
x0_sim = np.array([2.0, 0.0])  # Initial state

X_sim = [x0_sim]
U_sim = []

# --------------------------- Run MPC ---------------------------
for idx in range(len(t_sim) - N):
    x = X_sim[-1]
    
    # Initialize control sequence
    u_init = np.zeros(N)
    
    # Define bounds for all control inputs in the sequence
    bounds = [(u_min, u_max)] * N
    
    # Define the MPC cost function
    def mpc_cost(u_sequence, current_state, reference):
        cost = 0
        x_pred = current_state.copy()
        
        for u in u_sequence:
            # Clip control input to bounds
            u = np.clip(u, u_min, u_max)
            
            # State cost
            error = x_pred - reference
            cost += error.T @ Q @ error
            
            # Control cost
            cost += R * u**2
            
            # Propagate state using the discretized nonlinear model
            # Since the model is nonlinear, we need to account for higher-degree terms
            x_aug = np.concatenate([x_pred, [u]])
            x_dot_pred = model.predict(x_aug.reshape(1, -1))[0]
            x_pred = x_pred + x_dot_pred * dt  # Euler integration
        
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
        u_opt = result.x[0]
    else:
        print(f"Optimization failed at time {t_sim[idx]:.2f}")
        u_opt = 0.0
    
    # Apply control input
    U_sim.append(u_opt)
    
    # Simulate system with control input
    x_next = odeint(
        mass_spring_damper_nonlinear, 
        x, 
        [t_sim[idx], t_sim[idx+1]], 
        args=(u_opt,)
    )[-1]
    X_sim.append(x_next)

# Simulate the system for free vibration (u = 0)
X_free = np.zeros((len(t_sim), n_states))
X_free[0] = x0_sim
for i in range(len(t_sim)-1):
    sol = odeint(
        mass_spring_damper_nonlinear, 
        X_free[i], 
        [t_sim[i], t_sim[i+1]], 
        args=(0.0,)
    )
    X_free[i+1] = sol[-1]

# Convert results to arrays
X_sim = np.array(X_sim)
U_sim = np.array(U_sim)

# --------------------------- Plot Results ---------------------------
plt.figure(figsize=(12, 8))

# Position plot
plt.subplot(3,1,1)
plt.plot(t_sim[:len(X_sim)], X_sim[:, 0], label='Position with Control')
plt.plot(t_sim[:len(X_free)], X_free[:, 0], '--', label='Free Vibration')
plt.plot(t_sim, np.zeros_like(t_sim), 'r-', label='Reference')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)

# Velocity plot
plt.subplot(3,1,2)
plt.plot(t_sim[:len(X_sim)], X_sim[:, 1], label='Velocity with Control')
plt.plot(t_sim[:len(X_free)], X_free[:, 1], '--', label='Free Vibration')
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
plt.axhline(y=u_max, color='r', linestyle='--', alpha=0.3)
plt.axhline(y=u_min, color='r', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

# --------------------------- Phase Space Plot ---------------------------
plt.figure(figsize=(6,6))
plt.plot(X_sim[:, 0], X_sim[:, 1], label='Trajectory with Control')
plt.plot(X_free[:, 0], X_free[:, 1], '--', label='Free Vibration')
plt.plot(x0_sim[0], x0_sim[1], 'r*', label='Initial Condition')
plt.xlabel('Position (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Phase Space')
plt.legend()
plt.grid(True)
plt.show()