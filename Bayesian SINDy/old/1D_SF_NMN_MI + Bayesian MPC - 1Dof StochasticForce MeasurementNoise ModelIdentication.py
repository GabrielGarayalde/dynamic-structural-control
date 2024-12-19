# SDOF Bayesian SINDy with MPC Control and Uncertainty Overlay
# Optimized with centralized, vectorized feature definitions and multiple initial conditions for robustness testing.

import numpy as np

from scipy.integrate import odeint
import emcee
from tqdm import tqdm
from scipy.optimize import minimize
from SDOF_plotting import plot_parameter_distributions, print_parameter_comparison, plot_true_vs_estimated_uncontrolled_for_ic, plot_mpc_results
from SDOF_feature_definitions import build_library, get_feature_names, compute_features_vectorized, compute_true_coeffs

# ------------------------------
# 2. Bayesian SINDy Class
# ------------------------------

class BayesianSINDy:
    def __init__(self, n_walkers=32, b=0.01):
        self.b = b
        self.n_walkers = n_walkers
        self.samples = None
        self.feature_names = []

    def log_prior(self, theta):
        theta_0 = theta[0]
        sigma_epsilon = theta[-1]
        coeffs = theta[1:-1]

        if sigma_epsilon <= 0:
            return -np.inf

        # Laplace prior on coeffs
        log_prior = -len(coeffs)*np.log(2*self.b) - np.sum(np.abs(coeffs))/self.b

        # Gaussian prior for theta_0
        mu_theta0 = 0.0
        sigma_theta0 = 1.0
        log_prior += -0.5*((theta_0 - mu_theta0)/sigma_theta0)**2 - np.log(sigma_theta0*np.sqrt(2*np.pi))

        # Cauchy prior for sigma_epsilon
        scale_sigma_epsilon = 1.0
        log_prior += -np.log(np.pi*scale_sigma_epsilon*(1+(sigma_epsilon/scale_sigma_epsilon)**2))

        return log_prior

    def log_likelihood(self, theta, X, X_dot, U=None):
        theta_0 = theta[0]
        sigma_epsilon = theta[-1]
        Theta_matrix, feature_names = build_library(X, U=U)
        M = Theta_matrix.shape[1]
        coeffs_v = theta[1:1+M]

        # Predicted derivatives
        x_dot_pred = X[:, 1]
        v_dot_pred = theta_0 + Theta_matrix @ coeffs_v
        sigma_x = 1e-8  # Essentially fixed
        sigma_v = sigma_epsilon

        def normal_loglike(y, y_pred, sigma):
            return -0.5 * np.sum(((y - y_pred) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

        log_like_x = normal_loglike(X_dot[:,0], x_dot_pred, sigma_x)
        log_like_v = normal_loglike(X_dot[:,1], v_dot_pred, sigma_v)
        return log_like_x + log_like_v

    def log_probability(self, theta, X, X_dot, U=None):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta, X, X_dot, U)
        return lp + ll

    def fit(self, X, X_dot, U=None, n_steps=2000):
        Theta_matrix, feature_names = build_library(X, U=U)
        self.feature_names = feature_names
        M = Theta_matrix.shape[1]

        n_params = M + 2  # theta_0, coeffs_v (M), sigma_epsilon
        self.n_params = n_params

        initial_guess = np.zeros(n_params)
        initial_guess[-1] = 0.1  # sigma_epsilon > 0

        pos = initial_guess + 0.01 * np.random.randn(self.n_walkers, n_params)
        pos[:, -1] = np.abs(pos[:, -1]) + 1e-3  # Ensure sigma_epsilon > 0

        sampler = emcee.EnsembleSampler(
            self.n_walkers, n_params, self.log_probability,
            args=(X, X_dot, U)
        )

        print("Running MCMC (burn-in)...")
        state = sampler.run_mcmc(pos, 250, progress=True)
        sampler.reset()
        print("Running MCMC (production)...")
        sampler.run_mcmc(state, n_steps, progress=True)

        self.samples = sampler.get_chain(discard=500, thin=10, flat=True)
        return self

    def predict(self, x0, t, U=None, n_samples=100):
        if self.samples is None:
            raise ValueError("Model not fitted yet.")

        feature_names = get_feature_names()
        M = len(feature_names)

        idx = np.random.randint(len(self.samples), size=n_samples)
        predictions = []

        # Vectorized system function with noise
        def system(state, t_val, theta_params, u_val, noise):
            x, v = state
            theta_0 = theta_params[0]
            coeffs_v = theta_params[1:-1]  # Exclude sigma_epsilon

            # Compute features
            Theta_point = compute_features_vectorized(np.array([[x, v]]), np.array([u_val]))
            v_dot = theta_0 + Theta_point @ coeffs_v + noise  # Add noise to v_dot
            x_dot = v
            return [x_dot[0], v_dot[0]]

        print("Generating predictions:")
        for theta in tqdm(self.samples[idx], desc="Generating predictions"):
            sigma_epsilon = theta[-1]

            # Generate noise for all time steps
            noise_force = np.random.normal(0, sigma_epsilon, size=len(t))

            # Initialize prediction array
            X_pred = np.zeros((len(t), 2))
            X_pred[0] = x0

            for i in range(len(t)-1):
                # Current control input
                u_val = U[i] if U is not None else 0.0

                # Integrate over the current time step with noise
                sol = odeint(system, X_pred[i], [t[i], t[i+1]], args=(theta, u_val, noise_force[i]))
                X_pred[i+1] = sol[-1]

            predictions.append(X_pred)

        return np.array(predictions)



# ------------------------------
# 4. Additional Simulation Functions for Robustness
# ------------------------------

def identified_model_step_estimated(x, u_val, dt, theta_0, coeffs_v):
    """
    Perform one step of state propagation using the identified model parameters.
    """
    x_current, v_current = x
    # Compute derivatives
    v_dot = theta_0 + np.dot(coeffs_v, [x_current, v_current, x_current**2, v_current**2, x_current * v_current, u_val])
    # Update states using Euler's method
    x_next = x_current + dt * v_current
    v_next = v_current + dt * v_dot
    return np.array([x_next, v_next])

def simulate_true(m, c, k, theta_0, x0, t, U, sigma_epsilon):
    """
    Simulate the true system dynamics with control input U and process noise.
    
    Parameters:
    - m: Mass
    - c: Damping coefficient
    - k: Spring constant
    - theta_0: Bias term
    - x0: Initial state [x, v]
    - t: Time vector
    - U: Control input vector
    - sigma_epsilon: Standard deviation of process noise
    """
    def structural_dynamics(state, t_val, u_val, noise):
        x, v = state
        dx = v
        dv = (theta_0 + u_val - c * v - k * x) / m + noise  # Add noise to dv
        return [dx, dv]

    X_true = np.zeros((len(t), 2))
    X_true[0] = x0
    for i in range(len(t)-1):
        # Generate noise for the current time step
        noise = np.random.normal(0, sigma_epsilon)
        sol = odeint(structural_dynamics, X_true[i], [t[i], t[i+1]], args=(U[i], noise))
        X_true[i+1] = sol[-1]
    return X_true

def simulate_true_uncontrolled(m, c, k, theta_0, x0, t, sigma_epsilon):
    """
    Simulate the true system dynamics under u=0.
    """
    U_no_control = np.zeros(len(t))
    return simulate_true(m, c, k, theta_0, x0, t, U_no_control, sigma_epsilon)

def simulate_estimated_parameters_uncontrolled(model, t, x0, n_samples=50):
    """
    Simulate the system with u=0 using estimated parameters sampled from the posterior.
    """
    dt = t[1] - t[0]
    sampled_params = model.samples[np.random.choice(len(model.samples), size=n_samples, replace=False)]
    simulations = []

    feature_names = get_feature_names()
    M = len(feature_names)
    
    print("Simulating system under estimated parameters (u=0) for robustness...")
    for s in tqdm(range(n_samples), desc="Estimated Params Sim"):
        params = sampled_params[s]
        theta_0_id = params[0]
        coeffs_v_id = params[1:-1]

        x_current = x0.copy()
        X_sim = [x_current]

        for i in range(len(t)-1):
            u_val = 0.0  # u=0 for uncontrolled
            x_next = identified_model_step_estimated(x_current, u_val, dt, theta_0_id, coeffs_v_id)
            X_sim.append(x_next)
            x_current = x_next

        simulations.append(np.array(X_sim))

    simulations = np.array(simulations)
    return simulations


# ------------------------------
# 5. Model Predictive Control with Bayesian Uncertainty
# ------------------------------

def run_bayesian_mpc(model, x0, t, Q, R, N, u_min, u_max, x_ref, n_mpc_samples=50):
    """
    Run MPC with uncertainty by sampling from the posterior.
    For each sampled parameter set, run MPC and gather trajectories.
    """
    dt = t[1] - t[0]

    # Draw parameter samples
    sampled_params = model.samples[np.random.choice(len(model.samples), size=n_mpc_samples, replace=False)]

    # Arrays to store results
    X_all = np.zeros((n_mpc_samples, len(t), 2))
    U_all = np.zeros((n_mpc_samples, len(t)-1))  # len(t)-1 because control is applied at each step except the last

    # MPC cost function
    def mpc_cost(u_sequence, current_state, theta_0_id, coeffs_v_id, horizon):
        # Compute cost over the prediction horizon
        x_pred = current_state.copy()
        cost = 0.0
        for u_k in u_sequence[:horizon]:
            error = x_pred - x_ref
            cost += error.T @ Q @ error
            cost += R * (u_k ** 2)
            # Predict next state
            x_pred = identified_model_step_estimated(x_pred, u_k, dt, theta_0_id, coeffs_v_id)
        return cost

    print("Running Bayesian MPC simulations...")
    for s in tqdm(range(n_mpc_samples), desc="MPC Sim"):
        params = sampled_params[s]
        theta_0_id = params[0]
        coeffs_v_id = params[1:-1]

        x_current = x0.copy()
        X_sim = [x_current]
        U_sim = []
        for idx in range(len(t)-1):
            # Determine remaining steps
            remaining_steps = len(t) - idx - 1
            current_horizon = min(N, remaining_steps)

            # Define MPC optimization problem
            def cost_function(u_sequence):
                return mpc_cost(u_sequence, x_current, theta_0_id, coeffs_v_id, current_horizon)

            # Initial guess for control sequence
            u_init = np.zeros(current_horizon)

            # Bounds for control inputs
            bounds = [(u_min, u_max)] * current_horizon

            # Optimize control sequence
            res = minimize(
                cost_function,
                u_init,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-4}
            )

            if res.success:
                u_opt = res.x[0]
            else:
                u_opt = 0.0  # Fallback control
                print(f"Optimization failed at time {t[idx]:.2f}")

            U_sim.append(u_opt)
            x_next = identified_model_step_estimated(x_current, u_opt, dt, theta_0_id, coeffs_v_id)
            X_sim.append(x_next)
            x_current = x_next

        X_sim = np.array(X_sim)
        U_sim = np.array(U_sim)

        X_all[s, :len(X_sim), :] = X_sim
        U_all[s, :len(U_sim)] = U_sim

    # Compute statistics
    X_mean = np.mean(X_all, axis=0)
    X_std = np.std(X_all, axis=0)
    U_mean = np.mean(U_all, axis=0)
    U_std = np.std(U_all, axis=0)

    return X_all, U_all, X_mean, X_std, U_mean, U_std



# ------------------------------
# 7. Main Execution Block
# ------------------------------

if __name__ == "__main__":
    # ------------------------------
    # 7.1. System Parameters
    # ------------------------------
    m = 1.0              # Mass
    k = 1.0              # Spring constant
    c = 0.1              # Damping coefficient
    theta_0 = 0.5        # Bias term
    sigma_epsilon = 0.05 # Noise standard deviation

    dt = 0.005           # Time step
    t = np.arange(0, 20, dt)  # Time vector
    np.random.seed(42)    # For reproducibility

    # ------------------------------
    # 7.2. Control Input Definition
    # ------------------------------
    # Example Control Input: Sinusoidal
    u = 0.5 * np.sin(2 * np.pi * 0.5 * t)

    # ------------------------------
    # 7.3. System Dynamics Simulation with Control and Noise
    # ------------------------------
    # Initial conditions for robustness test
    initial_conditions = [
        [1.0, 0.0],    # Original initial condition
        [0.5, -0.2],    # Additional initial condition
        [1.5, 0.2]      # Another initial condition
    ]

    # Select the first initial condition for system simulation with control
    x0 = initial_conditions[0]

    # Simulate system dynamics with control input u and process noise
    X = simulate_true(m, c, k, theta_0, x0, t, u, sigma_epsilon)

    # Compute derivatives (X_dot) with noise
    X_dot = np.zeros((len(t), 2))
    for i in range(len(t)-1):
        # Compute derivatives from noisy dynamics
        dx = X[i,1]
        dv = (theta_0 + u[i] - c * X[i,1] - k * X[i,0]) / m + np.random.normal(0, sigma_epsilon)
        X_dot[i] = [dx, dv]
    X_dot[-1] = X_dot[-2]  # Assume last derivative is same as previous

    # ------------------------------
    # 7.4. Compute True Coefficients (For Comparison)
    # ------------------------------
    feature_names_example = get_feature_names()
    true_coeffs = compute_true_coeffs(m, k, c, theta_0, sigma_epsilon, feature_names_example)

    # ------------------------------
    # 7.5. Fit Bayesian SINDy Model
    # ------------------------------
    model = BayesianSINDy(n_walkers=16, b=0.1)
    model.fit(X, X_dot, U=u, n_steps=1000)

    # ------------------------------
    # 7.6. Inspect Predictions and Parameters (Optional)
    # ------------------------------
    # Generate predictions (optional, can be time-consuming)
    # predictions = model.predict(x0, t, U=u, n_samples=50)

    # Plot parameter distributions
    plot_parameter_distributions(model)

    # Print parameter comparison
    print_parameter_comparison(true_coeffs, model, feature_names_example)

    # ------------------------------
    # 7.7. Define MPC Parameters
    # ------------------------------
    N = 10                      # Prediction horizon
    Q = np.diag([100, 1])       # State weights: penalize position more than velocity
    R = 0.1                     # Control weight: penalize large control inputs
    u_max = 1.0                # Maximum control input
    u_min = -1.0               # Minimum control input
    x_ref = np.array([0.0, 0.0])  # Reference state: rest position and zero velocity

    # ------------------------------
    # 7.8. Run Bayesian MPC with Uncertainty
    # ------------------------------
    X_all, U_all, X_mean, X_std, U_mean, U_std = run_bayesian_mpc(
        model, x0, t, Q, R, N, u_min, u_max, x_ref, n_mpc_samples=5
    )

    # ------------------------------
    # 7.9. Simulate the Uncontrolled System (u = 0) for All Initial Conditions
    # ------------------------------
    for idx, ic in enumerate(initial_conditions):
        # Simulate true system under u=0 with process noise
        X_true_ic = simulate_true_uncontrolled(m, c, k, theta_0, ic, t, sigma_epsilon)
        
        # Compute derivatives with noise
        X_dot_ic = np.zeros((len(t), 2))
        for i in range(len(t)-1):
            dx = X_true_ic[i,1]
            dv = (theta_0 + 0 - c * X_true_ic[i,1] - k * X_true_ic[i,0]) / m + np.random.normal(0, sigma_epsilon)
            X_dot_ic[i] = [dx, dv]
        X_dot_ic[-1] = X_dot_ic[-2]
        
        # Simulate estimated system under u=0
        X_estimated_ic = simulate_estimated_parameters_uncontrolled(model, t, ic, n_samples=25)
        
        # Plot comparison
        plot_true_vs_estimated_uncontrolled_for_ic(t, X_true_ic, X_estimated_ic, initial_condition_label=f"IC{idx+1}")

    # ------------------------------
    # 7.10. Plot MPC Results with Uncertainty and Overlay Uncontrolled Trajectory
    # ------------------------------
    # Simulate uncontrolled system using the first initial condition
    X_uncontrolled = simulate_true_uncontrolled(m, c, k, theta_0, x0, t, sigma_epsilon)
    
    # Compute derivatives with noise
    X_dot_uncontrolled = np.zeros((len(t), 2))
    for i in range(len(t)-1):
        dx = X_uncontrolled[i,1]
        dv = (theta_0 + 0 - c * X_uncontrolled[i,1] - k * X_uncontrolled[i,0]) / m + np.random.normal(0, sigma_epsilon)
        X_dot_uncontrolled[i] = [dx, dv]
    X_dot_uncontrolled[-1] = X_dot_uncontrolled[-2]

    # Plot MPC results
    plot_mpc_results(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
