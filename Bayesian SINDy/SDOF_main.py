# SDOF Bayesian SINDy with MPC Control and Uncertainty Overlay
# Optimized with centralized, vectorized feature definitions and multiple initial conditions for robustness testing.

import numpy as np

from scipy.integrate import solve_ivp
import emcee
from tqdm import tqdm
from SDOF_plotting import (
    plot_parameter_distributions,
    print_parameter_comparison,
    plot_true_vs_estimated_uncontrolled_for_ic,
    plot_mpc_results
)
from SDOF_feature_definitions import (
    build_library,
    get_feature_names,
    compute_features_vectorized,
    compute_true_coeffs
)

from SDOF_bayesian_mpc import run_bayesian_mpc

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

        def system(t_val, state, theta_params, u_val, noise):
            x, v = state
            theta_0 = theta_params[0]
            coeffs_v = theta_params[1:-1]  # Exclude sigma_epsilon
        
            # Compute features
            Theta_point = compute_features_vectorized(np.array([[x, v]]), np.array([u_val]))
            # Extract the scalar result from the (1,) array
            v_dot_scalar = (Theta_point @ coeffs_v)[0]
            v_dot = theta_0 + v_dot_scalar + noise  # Now v_dot is a scalar
        
            x_dot = v  # v is already a scalar
            return [x_dot, v_dot]  # Both x_dot and v_dot are scalars now


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
                sol = solve_ivp(
                    system,
                    [t[i], t[i+1]],
                    X_pred[i],
                    args=(theta, u_val, noise_force[i]),
                    method='RK45',
                    t_eval=[t[i+1]]
                )

                if not sol.success:
                    raise RuntimeError(f"Integration failed at step {i}: {sol.message}")

                X_pred[i+1] = sol.y[:, -1]

            predictions.append(X_pred)

        return np.array(predictions)



# ------------------------------
# 4. Additional Simulation Functions for Robustness
# ------------------------------

def simulate_true(m, c, k, theta_0, x0, t, U, noise_array):
    """
    Simulate the true system dynamics with control input U and pre-generated noise array using solve_ivp.

    Parameters:
    - m: Mass
    - c: Damping coefficient
    - k: Spring constant
    - theta_0: Bias term
    - x0: Initial state [x, v] as a NumPy array
    - t: Time vector (sorted, strictly increasing)
    - U: Control input vector (same length as t)
    - noise_array: Pre-generated noise values for each time step (same length as t)

    Returns:
    - X_true: Array of shape (len(t), 2) with the states [x, v] at each time in t.
    """
    # Initialize state array
    X_true = np.zeros((len(t), 2))
    X_true[0] = x0

    # Define the ODE function, which uses u and noise from noise_array
    def dynamics(t_val, state, u_val, noise):
        x, v = state
        dxdt = v
        dvdt = (theta_0 + u_val - c * v - k * x) / m + noise
        return [dxdt, dvdt]

    # Integrate step-by-step using solve_ivp
    for i in range(len(t)-1):
        t_span = [t[i], t[i+1]]
        t_eval = [t[i+1]]
        u_val = U[i]
        noise = noise_array[i]

        sol = solve_ivp(
            dynamics,
            t_span,
            X_true[i],
            args=(u_val, noise),
            method='RK45',
            t_eval=t_eval
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed at step {i}: {sol.message}")

        X_true[i+1] = sol.y[:, -1]

    return X_true




# ------------------------------
# 7. Main Execution Block
# ------------------------------

if __name__ == "__main__":
    # ------------------------------
    # 7.1. System Parameters
    # ------------------------------
    m = 1.0              # Mass
    k = 1.0              # Spring constant
    c = 0.3              # Damping coefficient
    theta_0 = 0.5        # Bias term
    sigma_epsilon = 0.1 # Noise standard deviation

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
        np.array([1.0, 0.0]),    # Original initial condition
        np.array([0.5, -0.2]),   # Additional initial condition
        np.array([1.5, 0.2])     # Another initial condition
    ]

    # Select the first initial condition for system simulation with control
    x0 = initial_conditions[0]

    # Pre-generate noise array
    noise_array = np.random.normal(0, sigma_epsilon, size=len(t))


    # ------------------------------
    # DATA GENERATION
    # ------------------------------
    # Simulate system dynamics with control input u and process noise using solve_ivp
    X = simulate_true(m, c, k, theta_0, x0, t, u, noise_array)

    # Compute derivatives (X_dot) with the same pre-generated noise
    X_dot = np.zeros((len(t), 2))
    for i in range(len(t)-1):
        # Compute derivatives using pre-generated noise
        x, v = X[i]
        dx = v
        dv = (theta_0 + u[i] - c * v - k * x) / m + noise_array[i]
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
    model = BayesianSINDy(n_walkers=32, b=0.1)
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
    # 7.8. Simulate the Uncontrolled System (u = 0) for All Initial Conditions
    # ------------------------------
    for idx, ic in enumerate(initial_conditions):
        # Pre-generate noise array for uncontrolled simulation
        noise_array_ic = np.random.normal(0, sigma_epsilon, size=len(t))
        
        # Define control input array with u=0
        U_no_control = np.zeros(len(t))
        
        # Simulate true system under u=0 with process noise using simulate_true
        X_true_ic = simulate_true(m, c, k, theta_0, ic, t, U_no_control, noise_array_ic)

        # Compute derivatives with pre-generated noise
        X_dot_ic = np.zeros((len(t), 2))
        for i in range(len(t)-1):
            x, v = X_true_ic[i]
            dx = v
            dv = (theta_0 + 0 - c * v - k * x) / m + noise_array_ic[i]
            X_dot_ic[i] = [dx, dv]
        X_dot_ic[-1] = X_dot_ic[-2]
        

        # Use the model's predict method to generate simulations
        X_estimated_ic = model.predict(ic, t, U=U_no_control, n_samples=5)
                
        # Plot comparison
        plot_true_vs_estimated_uncontrolled_for_ic(t, X_true_ic, X_estimated_ic, initial_condition_label=f"IC{idx+1}")
    
    
    # ------------------------------
    # 7.7. Define MPC Parameters
    # ------------------------------
    N = 20                      # Prediction horizon
    Q = np.diag([100, 1])       # State weights: penalize position more than velocity
    R = 0.1                     # Control weight: penalize large control inputs
    u_max = 1.0                # Maximum control input
    u_min = -1.0               # Minimum control input
    x_ref = np.array([0.0, 0.0])  # Reference state: rest position and zero velocity


    # ------------------------------
    # 7.9. Run Bayesian MPC with Uncertainty
    # ------------------------------
    X_all, U_all, X_mean, X_std, U_mean, U_std = run_bayesian_mpc(
        model, x0, t, Q, R, N, u_min, u_max, x_ref, n_mpc_samples=3
    )
    
    # ------------------------------
    # 7.10. Plot MPC Results with Uncertainty and Overlay Uncontrolled Trajectory
    # ------------------------------
    # Pre-generate noise array for uncontrolled simulation
    noise_array_uncontrolled = np.random.normal(0, sigma_epsilon, size=len(t))
    
    # Define control input array with u=0
    U_no_control = np.zeros(len(t))
    
    # Simulate uncontrolled system using the first initial condition
    X_uncontrolled = simulate_true(m, c, k, theta_0, x0, t, U_no_control, noise_array_uncontrolled)
    
    # Compute derivatives with pre-generated noise
    X_dot_uncontrolled = np.zeros((len(t), 2))
    for i in range(len(t)-1):
        x, v = X_uncontrolled[i]
        dx = v
        dv = (theta_0 + 0 - c * v - k * x) / m + noise_array_uncontrolled[i]
        X_dot_uncontrolled[i] = [dx, dv]
    X_dot_uncontrolled[-1] = X_dot_uncontrolled[-2]

    # Plot MPC results
    plot_mpc_results(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
