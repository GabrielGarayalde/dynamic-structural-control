import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import emcee
import corner
from tqdm import tqdm

class BayesianSINDy:
    def __init__(self, n_params=3, n_walkers=32, b=0.01, measurement_noise_std=0.01, dt=0.005):
        self.n_params = n_params
        self.n_walkers = n_walkers
        self.samples = None
        self.feature_names = ['Stiffness', 'Damping', 'Sigma_v']
        self.b = b  # Sparsity parameter
        self.measurement_noise_std = measurement_noise_std
        self.dt = dt  # Time step


    
    def log_prior(self, theta):
        K, C, sigma_v = theta
        # Physical constraints and other priors
        # Physical constraints
        if not (0 < K < 5.0 and 0 < C < 1.0 and sigma_v >= 0):
            return -np.inf  # Outside physical bounds
        
        # Priors
        log_prior = 0
        
        # Laplace prior for sparsity (K, C)
        log_prior += -np.log(2 * self.b) - np.abs(K) / self.b
        log_prior += -np.log(2 * self.b) - np.abs(C) / self.b
    
        # Adjust scale for sigma_v prior
        scale_sigma_v = self.measurement_noise_std / dt
        if sigma_v <= 0:
            return -np.inf  # sigma_v must be positive
        log_prior += -np.log(np.pi * scale_sigma_v * (1 + (sigma_v / scale_sigma_v) ** 2))
    
        return log_prior


    def log_likelihood(self, theta, X, X_dot):
        K, C, sigma_v = theta
    
        # Predicted derivatives from the model
        x_dot_pred = X[:, 1]  # dx/dt = v
        v_dot_pred = -K * X[:, 0] - C * X[:, 1]
    
        # Measurement noise standard deviations
        sigma_x = self.measurement_noise_std  # Position measurement noise
    
        # Adjust sigma_v for derivative calculation
        sigma_v_effective = sigma_v / dt
    
        # Compute log-likelihood for position and velocity derivatives
        log_like_x = -0.5 * np.sum(((X_dot[:, 0] - x_dot_pred) / sigma_x)**2 + np.log(2 * np.pi * sigma_x**2))
        log_like_v = -0.5 * np.sum(((X_dot[:, 1] - v_dot_pred) / sigma_v_effective)**2 + np.log(2 * np.pi * sigma_v_effective**2))
    
        return log_like_x + log_like_v


    def log_probability(self, theta, X, X_dot):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta, X, X_dot)
        return lp + ll

    def fit(self, X, X_dot, n_steps=5000):
        """
        Perform MCMC sampling.
        """
        initial_guess = np.array([1.0, 0.1, self.measurement_noise_std / dt])
        # Initialize walkers
        pos = initial_guess + 0.1 * np.random.randn(self.n_walkers, self.n_params)


        # Ensure parameters are within bounds
        pos[:, 0] = np.abs(pos[:, 0])  # Stiffness K (positive)
        pos[:, 1] = np.abs(pos[:, 1])  # Damping C (positive)
        pos[:, 2] = np.abs(pos[:, 2])  # sigma_v (positive)

        # Initialize and run the sampler
        sampler = emcee.EnsembleSampler(
            self.n_walkers, self.n_params, self.log_probability,
            args=(X, X_dot)
        )

        print("Running MCMC...")
        # Burn-in phase
        state = sampler.run_mcmc(pos, 1000, progress=True)
        sampler.reset()
        # Sampling phase
        sampler.run_mcmc(state, n_steps, progress=True)

        # Thinning and flattening the chain
        self.samples = sampler.get_chain(discard=500, thin=10, flat=True)

        return self

    def predict(self, x0, t, n_samples=100):
        """
        Generate predictions with uncertainty by sampling from posterior.
        """
        predictions = []
        idx = np.random.randint(len(self.samples), size=n_samples)
        theta_samples = self.samples[idx]
    
        for theta in tqdm(theta_samples, desc="Generating predictions"):
            K, C, sigma_v = theta
            X_pred = np.zeros((len(t), 2))
            X_pred[0] = x0
    
            # Generate predictions for the entire time series
            for i in range(len(t) - 1):
                def system(state, t_val):
                    x, v = state
                    dx = v
                    dv = -K * x - C * v  # No forcing term
                    return [dx, dv]
    
                sol = odeint(system, X_pred[i], [t[i], t[i + 1]])
                X_pred[i + 1] = sol[-1]
    
            predictions.append(X_pred)
    
        return np.array(predictions)




def plot_true_vs_predicted(t, X_true, predictions):
    """
    Plot true vs. predicted states with confidence intervals.
    """
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Position
    axs[0].plot(t, X_true[:, 0], 'b-', label='True Position')
    axs[0].plot(t, pred_mean[:, 0], 'r--', label='Mean Prediction')
    axs[0].fill_between(t, pred_mean[:, 0] - 2 * pred_std[:, 0], pred_mean[:, 0] + 2 * pred_std[:, 0],
                        color='r', alpha=0.2, label='95% CI')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Position')
    axs[0].legend()
    axs[0].set_title('Position with Uncertainty')
    axs[0].grid(True)

    # Velocity
    axs[1].plot(t, X_true[:, 1], 'b-', label='True Velocity')
    axs[1].plot(t, pred_mean[:, 1], 'r--', label='Mean Prediction')
    axs[1].fill_between(t, pred_mean[:, 1] - 2 * pred_std[:, 1], pred_mean[:, 1] + 2 * pred_std[:, 1],
                        color='r', alpha=0.2, label='95% CI')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Velocity')
    axs[1].legend()
    axs[1].set_title('Velocity with Uncertainty')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_parameter_distributions(model, param_names):
    """
    Plot posterior distributions of parameters.
    """
    fig = corner.corner(model.samples,
                        labels=param_names,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12})
    plt.show()

def print_parameter_comparison(true_values, model, param_names):
    """
    Compare true coefficients with estimated parameters and report errors.
    """
    print("\n### Parameter Comparison ###\n")

    def format_comparison(true_val, estimates, label):
        median = np.median(estimates)
        std = np.std(estimates)
        if true_val == 0:
            absolute_error = np.abs(median)
            return f"{label:15}: True = {true_val:.5f}, Estimate = {median:.5f} ± {std:.5f}, Abs. Error = {absolute_error:.5f}"
        else:
            percentage_error = 100 * np.abs(median - true_val) / np.abs(true_val)
            return f"{label:15}: True = {true_val:.5f}, Estimate = {median:.5f} ± {std:.5f}, Error = {percentage_error:.2f}%"

    for i, name in enumerate(param_names):
        print(format_comparison(true_values[i], model.samples[:, i], name))

def structural_dynamics(state, t):
    """
    Simplified system dynamics without forcing or process noise.
    """
    M = 1.0  # Mass
    K = 1.0  # Stiffness
    C = 0.1  # Damping

    x, v = state
    acceleration = (-K * x - C * v) / M

    return [v, acceleration]

# Main execution
if __name__ == "__main__":
    # System parameters and data generation
    dt = 0.01
    t = np.arange(0, 20, dt)

    # Initial conditions
    x0 = [1.0, 0.0]  # Initial displacement and velocity

    # Generate true system response
    X = np.zeros((len(t), 2))
    X[0] = x0

    for i in range(len(t) - 1):
        sol = odeint(structural_dynamics, X[i], [t[i], t[i + 1]])
        X[i + 1] = sol[-1]

    # Add measurement noise
    measurement_noise_std = 0.005  # Small measurement noise
    measurement_noise = np.random.normal(0, measurement_noise_std, size=X.shape)
    X_noisy = X + measurement_noise

    # Compute derivatives using finite differences
    from scipy.signal import savgol_filter

    # Apply Savitzky-Golay filter to smooth data
    window_length = 51  # Must be odd
    polyorder = 3
    
    X_smooth = savgol_filter(X_noisy, window_length, polyorder, axis=0)
    # Compute derivatives on smoothed data
    X_dot = np.gradient(X_smooth, dt, axis=0)

    # X_dot = np.gradient(X_noisy, dt, axis=0)

    # Fit Bayesian SINDy model
    model = BayesianSINDy(n_params=3, b=0.01, measurement_noise_std=measurement_noise_std, dt=dt)
    model.fit(X_noisy, X_dot, n_steps=5000)

    # Generate predictions
    predictions = model.predict(x0, t)

    # Plot results
    plot_true_vs_predicted(t, X, predictions)
    plot_parameter_distributions(model, model.feature_names)

    # Compare parameters
    true_coeffs = [1.0, 0.1, measurement_noise_std]
    print_parameter_comparison(true_coeffs, model, model.feature_names)

    # Robustness test with different initial conditions
    print("\n### Robustness Test: Different Initial Conditions ###")
    new_x0 = [0.5, -0.5]  # New initial displacement and velocity
    X_new = np.zeros((len(t), 2))
    X_new[0] = new_x0

    for i in range(len(t) - 1):
        sol = odeint(structural_dynamics, X_new[i], [t[i], t[i + 1]])
        X_new[i + 1] = sol[-1]

    # Add measurement noise to new initial condition response
    measurement_noise_new = np.random.normal(0, measurement_noise_std, size=X_new.shape)
    X_new_noisy = X_new + measurement_noise_new

    # Predict for new initial conditions
    predictions_new = model.predict(new_x0, t)

    # Plot true vs predicted states for new initial conditions
    plot_true_vs_predicted(t, X_new, predictions_new)

    # Compute RMSE for robustness evaluation
    rmse_x = np.sqrt(np.mean((X_new[:, 0] - np.mean(predictions_new[:, :, 0], axis=0)) ** 2))
    rmse_v = np.sqrt(np.mean((X_new[:, 1] - np.mean(predictions_new[:, :, 1], axis=0)) ** 2))

    print("\n### RMSE for New Initial Conditions ###")
    print(f"x RMSE: {rmse_x:.5f}")
    print(f"v RMSE: {rmse_v:.5f}")
