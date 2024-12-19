# TDOF_main.py

import numpy as np
from scipy.integrate import solve_ivp
import emcee
from tqdm import tqdm

from TwoDOF_feature_definitions import (
    build_library,
    get_feature_names,
    compute_features_vectorized,
    compute_true_coeffs
)
from TwoDOF_bayesian_mpc import run_bayesian_mpc

# Import plotting functions from your TDOF_plotting file
from TwoDOF_plotting import (
    plot_parameter_distributions,
    print_parameter_comparison,
    plot_true_vs_estimated_uncontrolled_for_ic,
    plot_mpc_results
)

from TwoDOF_pysindy_initial_guess import get_initial_guess_from_pysindy
# from test_script import get_initial_guess_from_pysindy

# ------------------------------
# Bayesian SINDy Class for 2DOF
# ------------------------------

class BayesianSINDy:
    def __init__(self, n_walkers=32, b=0.01):
        self.b = b
        self.n_walkers = n_walkers
        self.samples = None
        self.feature_names = []

    def log_prior(self, theta):
        # Parameter vector structure:
        # [theta_0_1, coeffs_1..., theta_0_2, coeffs_2..., sigma_epsilon_1, sigma_epsilon_2]
        M = len(self.feature_names)
        theta_0_1 = theta[0]
        coeffs_1 = theta[1:1+M]
        theta_0_2 = theta[1+M]
        coeffs_2 = theta[2+M:2+2*M]
        sigma_epsilon_1 = theta[-2]
        sigma_epsilon_2 = theta[-1]

        if sigma_epsilon_1 <= 0 or sigma_epsilon_2 <= 0:
            return -np.inf

        # Laplace prior on coeffs
        all_coeffs = np.concatenate([coeffs_1, coeffs_2])
        log_prior = -len(all_coeffs)*np.log(2*self.b) - np.sum(np.abs(all_coeffs))/self.b

        # Gaussian priors for theta_0_1 and theta_0_2
        mu_theta0 = 0.0
        sigma_theta0 = 1.0
        for t0 in [theta_0_1, theta_0_2]:
            log_prior += -0.5*((t0 - mu_theta0)/sigma_theta0)**2 - np.log(sigma_theta0*np.sqrt(2*np.pi))

        # Cauchy prior for sigma_epsilon_1 and sigma_epsilon_2
        scale_sigma_epsilon = 1.0
        for se in [sigma_epsilon_1, sigma_epsilon_2]:
            log_prior += -np.log(np.pi*scale_sigma_epsilon*(1+(se/scale_sigma_epsilon)**2))

        return log_prior

    def log_likelihood(self, theta, X, X_dot, U=None):
        M = len(self.feature_names)
        theta_0_1 = theta[0]
        coeffs_1 = theta[1:1+M]
        theta_0_2 = theta[1+M]
        coeffs_2 = theta[2+M:2+2*M]
        sigma_epsilon_1 = theta[-2]
        sigma_epsilon_2 = theta[-1]

        # Build library
        Theta_matrix, _ = build_library(X, U=U)
        # Predicted derivatives
        # States: X = [x1, v1, x2, v2]
        # True derivatives from X_dot: X_dot = [dx1, dv1, dx2, dv2]
        # dx1 = v1, dx2 = v2 by definition
        # We model v1_dot and v2_dot
        v1_dot_pred = theta_0_1 + Theta_matrix @ coeffs_1
        v2_dot_pred = theta_0_2 + Theta_matrix @ coeffs_2

        v1_dot_true = X_dot[:,1]
        v2_dot_true = X_dot[:,3]

        sigma_v1 = sigma_epsilon_1
        sigma_v2 = sigma_epsilon_2

        def normal_loglike(y, y_pred, sigma):
            return -0.5 * np.sum(((y - y_pred) / sigma)**2 + np.log(2*np.pi*sigma**2))

        # For x1_dot and x2_dot, we treat them as known with negligible noise
        sigma_x = 1e-8
        x1_dot_pred = X[:,1]  # = v1
        x2_dot_pred = X[:,3]  # = v2
        x1_dot_true = X_dot[:,0]
        x2_dot_true = X_dot[:,2]

        log_like_x1 = normal_loglike(x1_dot_true, x1_dot_pred, sigma_x)
        log_like_x2 = normal_loglike(x2_dot_true, x2_dot_pred, sigma_x)
        log_like_v1 = normal_loglike(v1_dot_true, v1_dot_pred, sigma_v1)
        log_like_v2 = normal_loglike(v2_dot_true, v2_dot_pred, sigma_v2)

        return log_like_x1 + log_like_x2 + log_like_v1 + log_like_v2

    def log_probability(self, theta, X, X_dot, U=None):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta, X, X_dot, U)
        return lp + ll

    def fit(self, X, X_dot, U=None, n_steps=2000, initial_guess_params=None):
        Theta_matrix, feature_names = build_library(X, U=U)
        self.feature_names = feature_names
        M = Theta_matrix.shape[1]
    
        # Parameter count:
        # [theta_0_1, coeffs_1..., theta_0_2, coeffs_2..., sigma_epsilon_1, sigma_epsilon_2]
        n_params = 2 + 2*M + 2
    
        if initial_guess_params is not None:
            # Use provided initial guess
            if len(initial_guess_params) != n_params:
                raise ValueError(f"Initial guess must have length {n_params}")
            initial_guess = initial_guess_params.copy()
        else:
            # Default initial guess
            initial_guess = np.zeros(n_params)
            initial_guess[-2] = 0.1
            initial_guess[-1] = 0.1
    
        pos = initial_guess + 0.01 * np.random.randn(self.n_walkers, n_params)
        pos[:, -2] = np.abs(pos[:, -2]) + 1e-3
        pos[:, -1] = np.abs(pos[:, -1]) + 1e-3
    
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
        """
        Predicts the system states over time using sampled parameters from the Bayesian SINDy model.
        Parameters:
            x0: Initial state [x1, v1, x2, v2]
            t: Time array
            U: Control input array, same length as t (or None for zero input)
            n_samples: Number of parameter samples to draw from the posterior
        Returns:
            predictions: Array of shape (n_samples, len(t), 4) containing predicted states.
        """
        if self.samples is None:
            raise ValueError("Model not fitted yet.")
    
        feature_names = self.feature_names
        M = len(feature_names)
    
        # Randomly select parameter samples
        idx = np.random.randint(len(self.samples), size=n_samples)
        predictions = []
    
        # System ODE given a parameter set and a control input
        def system(t_val, state, theta_params, u_val, noise_1, noise_2):
            x1, v1, x2, v2 = state
            theta_0_1 = theta_params[0]
            coeffs_1 = theta_params[1:1+M]
            theta_0_2 = theta_params[1+M]
            coeffs_2 = theta_params[2+M:2+2*M]
    
            # Compute features
            X_point = np.array([[x1, v1, x2, v2]])
            U_point = np.array([u_val])
            Theta_point = compute_features_vectorized(X_point, U_point)
    
            v1_dot = theta_0_1 + (Theta_point @ coeffs_1)[0] + noise_1
            v2_dot = theta_0_2 + (Theta_point @ coeffs_2)[0] + noise_2
    
            dx1 = v1
            dx2 = v2
            dv1 = v1_dot
            dv2 = v2_dot
            return [dx1, dv1, dx2, dv2]
    
        print("Generating predictions:")
        for theta in tqdm(self.samples[idx], desc="Generating predictions"):
            sigma_epsilon_1 = theta[-2]
            sigma_epsilon_2 = theta[-1]
    
            # Generate noise for all time steps for v1_dot and v2_dot
            noise_force_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
            noise_force_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))
    
            # Initialize prediction array
            X_pred = np.zeros((len(t), 4))
            X_pred[0] = x0
    
            for i in range(len(t)-1):
                # Current control input
                u_val = U[i] if U is not None else 0.0
                # Integrate over the current time step with noise
                sol = solve_ivp(
                    system,
                    [t[i], t[i+1]],
                    X_pred[i],
                    args=(theta, u_val, noise_force_1[i], noise_force_2[i]),
                    method='RK45',
                    t_eval=[t[i+1]]
                )
    
                if not sol.success:
                    raise RuntimeError(f"Integration failed at step {i}: {sol.message}")
    
                X_pred[i+1] = sol.y[:, -1]
    
            predictions.append(X_pred)
    
        return np.array(predictions)

# ------------------------------
# True System Simulation
# ------------------------------

def simulate_true(m1, m2, c1, c2, k1, k2, theta_0_1, theta_0_2, x0, t, U, noise_array_1, noise_array_2):
    """
    Simulate the true 2DOF system dynamics. 
    X = [x1, v1, x2, v2]
    v1_dot = (theta_0_1 - c1*v1 - k1*x1 - c2*(v1 - v2) - k2*(x1 - x2))/m1 + noise_1
    v2_dot = (theta_0_2 + u - c2*(v2 - v1) - k2*(x2 - x1))/m2 + noise_2
    """

    X_true = np.zeros((len(t), 4))
    X_true[0] = x0

    def dynamics(t_val, state, u_val, n1, n2):
        x1, v1, x2, v2 = state
        dx1 = v1
        dx2 = v2
        dv1 = (theta_0_1 - c1*v1 - k1*x1 - c2*(v1 - v2) - k2*(x1 - x2))/m1 + n1
        dv2 = (theta_0_2 + u_val - c2*(v2 - v1) - k2*(x2 - x1))/m2 + n2
        return [dx1, dv1, dx2, dv2]

    for i in range(len(t)-1):
        dt_span = [t[i], t[i+1]]
        t_eval = [t[i+1]]
        u_val = U[i]
        n1 = noise_array_1[i]
        n2 = noise_array_2[i]

        sol = solve_ivp(
            dynamics,
            dt_span,
            X_true[i],
            args=(u_val, n1, n2),
            method='RK45',
            t_eval=t_eval
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed at step {i}: {sol.message}")

        X_true[i+1] = sol.y[:, -1]

    return X_true


if __name__ == "__main__":
    # ------------------------------
    # System Parameters
    # ------------------------------
    m1 = 1.0
    m2 = 1.0
    k1 = 1.0
    k2 = 1.0
    c1 = 0.3
    c2 = 0.3
    theta_0_1 = 0.5
    theta_0_2 = 0.5
    sigma_epsilon_1 = 0.1
    sigma_epsilon_2 = 0.1

    dt = 0.005
    t = np.arange(0, 20, dt)
    np.random.seed(42)

    # Control input on second mass
    u = 0.5 * np.sin(2 * np.pi * 0.5 * t)

    # Initial conditions
    x0 = np.array([1.0, 0.0, 0.5, -0.2])  # Example initial condition

    # Noise arrays
    noise_array_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
    noise_array_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))

    # ------------------------------
    # Simulate True System
    # ------------------------------
    X = simulate_true(m1, m2, c1, c2, k1, k2, theta_0_1, theta_0_2, x0, t, u, noise_array_1, noise_array_2)

    # Compute derivatives
    X_dot = np.zeros((len(t), 4))
    for i in range(len(t)-1):
        x1, v1, x2, v2 = X[i]
        dx1 = v1
        dx2 = v2
        dv1 = (theta_0_1 - c1*v1 - k1*x1 - c2*(v1 - v2) - k2*(x1 - x2))/m1 + noise_array_1[i]
        dv2 = (theta_0_2 + u[i] - c2*(v2 - v1) - k2*(x2 - x1))/m2 + noise_array_2[i]
        X_dot[i] = [dx1, dv1, dx2, dv2]
    X_dot[-1] = X_dot[-2]

    # ------------------------------
    # True Coefficients
    # ------------------------------
    feature_names_example = get_feature_names()
    true_coeffs = compute_true_coeffs(m1, m2, c1, c2, k1, k2, theta_0_1, theta_0_2, sigma_epsilon_1, sigma_epsilon_2, feature_names_example)

    # ------------------------------
    # Fit Bayesian SINDy Model
    # ------------------------------
    # In your main or wherever you call fit:
    # Compute initial guess with PySINDy
    initial_guess = get_initial_guess_from_pysindy(X, X_dot, u, t)

    model = BayesianSINDy(n_walkers=500, b=0.1)
    model.fit(X, X_dot, U=u, n_steps=1000, initial_guess_params=initial_guess)

    # ------------------------------
    # Inspect Parameters
    # ------------------------------
    plot_parameter_distributions(model)
    print_parameter_comparison(true_coeffs, model, feature_names_example)

    # ------------------------------
    # Simulate Uncontrolled System (u=0) for Comparison
    # ------------------------------
    # We'll reuse the same initial condition and times
    noise_array_uncontrolled_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
    noise_array_uncontrolled_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))
    U_no_control = np.zeros(len(t))
    X_uncontrolled = simulate_true(m1, m2, c1, c2, k1, k2, theta_0_1, theta_0_2, x0, t, U_no_control, noise_array_uncontrolled_1, noise_array_uncontrolled_2)

    # Compute derivatives for uncontrolled
    X_dot_uncontrolled = np.zeros((len(t), 4))
    for i in range(len(t)-1):
        x1, v1, x2, v2 = X_uncontrolled[i]
        dx1 = v1
        dx2 = v2
        dv1 = (theta_0_1 - c1*v1 - k1*x1 - c2*(v1 - v2) - k2*(x1 - x2))/m1 + noise_array_uncontrolled_1[i]
        dv2 = (theta_0_2 + 0 - c2*(v2 - v1) - k2*(x2 - x1))/m2 + noise_array_uncontrolled_2[i]
        X_dot_uncontrolled[i] = [dx1, dv1, dx2, dv2]
    X_dot_uncontrolled[-1] = X_dot_uncontrolled[-2]

    # Use the model to predict under uncontrolled conditions
    # For prediction, we can pick a small number of samples
    X_estimated_uncontrolled = model.predict(x0, t, U=U_no_control, n_samples=5)

    # Plot True vs Estimated for uncontrolled scenario
    plot_true_vs_estimated_uncontrolled_for_ic(t, X_uncontrolled, X_estimated_uncontrolled, "IC")

    # ------------------------------
    # MPC Parameters
    # ------------------------------
    N = 20
    Q = np.diag([100, 1, 100, 1])
    R = 0.1
    u_max = 1.0
    u_min = -1.0
    x_ref = np.array([0.0, 0.0, 0.0, 0.0])

    # ------------------------------
    # Run Bayesian MPC
    # ------------------------------
    X_all, U_all, X_mean, X_std, U_mean, U_std = run_bayesian_mpc(
        model, x0, t, Q, R, N, u_min, u_max, x_ref, n_mpc_samples=3
    )

    # ------------------------------
    # Plot MPC Results
    # ------------------------------
    plot_mpc_results(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
