import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import emcee
import corner
from tqdm import tqdm

class BayesianSINDy:
    def __init__(self, n_params=8, n_walkers=32, b=0.01):
        self.n_params = n_params
        self.n_walkers = n_walkers
        self.samples = None
        self.b = b  # Default value for sparsity parameter

        self.feature_names = ['Constant', 'Stiffness K', 'Damping C', 'Coeff x^2', 'Coeff v^2', 'Coeff xv', 'Sigma_epsilon', 'Sigma_v']

    def log_prior(self, theta):
        """
        Priors for the parameters with adjustable sparsity via the Laplace scale parameter b.
        """
        # Unpack parameters
        theta_0, K, C, alpha_x2, alpha_v2, alpha_xv, sigma_epsilon, sigma_v = theta
    
        # Physical constraints: Ensure parameters are within valid ranges
        if not (0 < K < 5.0 and 0 < C < 1.0 and sigma_epsilon > 0 and sigma_v > 0):
            return -np.inf  # Outside physical bounds
    
        # Initialize the log-prior
        log_prior = 0
    
        # Laplace prior for sparsity (K, C, alpha_x2, alpha_v2, alpha_xv)
        for param in [K, C, alpha_x2, alpha_v2, alpha_xv]:
            log_prior += -np.log(2 * self.b) - np.abs(param) / self.b
    
        # Gaussian prior for theta_0 (mean force)
        mu_theta0 = 0.0
        sigma_theta0 = 1.0
        log_prior += -0.5 * ((theta_0 - mu_theta0) / sigma_theta0) ** 2 - np.log(sigma_theta0 * np.sqrt(2 * np.pi))
    
        # Half-Cauchy prior for sigma_epsilon (process noise)
        scale_sigma_epsilon = 1.0
        log_prior += -np.log(np.pi * scale_sigma_epsilon * (1 + (sigma_epsilon / scale_sigma_epsilon) ** 2))
    
        # Half-Cauchy prior for sigma_v (measurement noise)
        scale_sigma_v = 0.1
        log_prior += -np.log(np.pi * scale_sigma_v * (1 + (sigma_v / scale_sigma_v) ** 2))
    
        return log_prior



    def log_likelihood(self, theta, X, X_dot):
        """
        Likelihood function accounting for unobserved process noise.
        """
        try:
            theta_0, K, C, alpha_x2, alpha_v2, alpha_xv, sigma_epsilon, sigma_v = theta

            # Predicted derivatives from the model
            x_dot_pred = X[:, 1]  # dx/dt = v

            # Include additional terms in the model
            v_dot_pred = (theta_0
                          - K * X[:, 0]
                          - C * X[:, 1]
                          - alpha_x2 * X[:, 0] ** 2
                          - alpha_v2 * X[:, 1] ** 2
                          - alpha_xv * X[:, 0] * X[:, 1])

            # Measurement noise standard deviations
            sigma_x = 0.05  # Position derivative noise

            # Total variance for dv/dt (measurement noise + process noise)
            sigma_total_v = np.sqrt(sigma_v ** 2 + sigma_epsilon ** 2)

            # Compute log-likelihood for position and velocity derivatives
            log_like_x = -0.5 * np.sum(((X_dot[:, 0] - x_dot_pred) / sigma_x) ** 2 + np.log(2 * np.pi * sigma_x ** 2))
            log_like_v = -0.5 * np.sum(((X_dot[:, 1] - v_dot_pred) / sigma_total_v) ** 2 + np.log(2 * np.pi * sigma_total_v ** 2))

            return log_like_x + log_like_v

        except:
            return -np.inf

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
        # Initial guess for parameters
        initial_guess = np.array([0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.5, 0.1])  # [theta_0, K, C, alpha_x2, alpha_v2, alpha_xv, sigma_epsilon, sigma_v]

        # Initialize walkers around the initial guess
        pos = initial_guess + 0.1 * np.random.randn(self.n_walkers, self.n_params)

        # Ensure parameters are within bounds
        pos[:, 1] = np.abs(pos[:, 1])  # Stiffness K (positive)
        pos[:, 2] = np.abs(pos[:, 2])  # Damping C (positive)
        pos[:, 6] = np.abs(pos[:, 6])  # sigma_epsilon (positive)
        pos[:, 7] = np.abs(pos[:, 7])  # sigma_v (positive)

        # Initialize and run the sampler
        sampler = emcee.EnsembleSampler(
            self.n_walkers, self.n_params, self.log_probability,
            args=(X, X_dot)
        )

        print("Running MCMC...")
        # Burn-in phase
        state = sampler.run_mcmc(pos, 2000, progress=True)
        sampler.reset()
        # Sampling phase
        sampler.run_mcmc(state, n_steps, progress=True)

        # Thinning and flattening the chain
        self.samples = sampler.get_chain(discard=1000, thin=20, flat=True)

        return self

    def predict(self, x0, t, n_samples=100):
        """
        Generate predictions with uncertainty by sampling from posterior.
        """
        predictions = []
        idx = np.random.randint(len(self.samples), size=n_samples)
        theta_samples = self.samples[idx]

        for theta in tqdm(theta_samples, desc="Generating predictions"):
            theta_0, K, C, alpha_x2, alpha_v2, alpha_xv, sigma_epsilon, _ = theta
            X_pred = np.zeros((len(t), 2))
            X_pred[0] = x0

            # Generate new noise sequence for each prediction
            np.random.seed()  # Ensure randomness
            noise_force = np.random.normal(0, sigma_epsilon, size=len(t))

            for i in range(len(t) - 1):
                def system(state, t_val):
                    x, v = state
                    dx = v
                    dv = (theta_0
                          - K * x
                          - C * v
                          - alpha_x2 * x ** 2
                          - alpha_v2 * v ** 2
                          - alpha_xv * x * v
                          + noise_force[i])
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
            return f"{label:20}: True = {true_val:.5f}, Estimate = {median:.5f} ± {std:.5f}, Abs. Error = {absolute_error:.5f}"
        else:
            percentage_error = 100 * np.abs(median - true_val) / np.abs(true_val)
            return f"{label:20}: True = {true_val:.5f}, Estimate = {median:.5f} ± {std:.5f}, Error = {percentage_error:.2f}%"

    for i, name in enumerate(param_names):
        print(format_comparison(true_values[i], model.samples[:, i], name))

# Define the true system dynamics for generating training data
def structural_dynamics(state, t, noise_force):
    M = 1.0  # Mass
    K = 1.0  # Stiffness
    C = 0.1  # Damping

    x, v = state
    acceleration = (-K * x - C * v + noise_force) / M

    return [v, acceleration]

# Main execution
if __name__ == "__main__":
    # System parameters and data generation
    dt = 0.005
    t = np.arange(0, 20, dt)
    np.random.seed(42)

    # True model parameters
    true_mass = 1.0
    true_stiffness = 1.0  # K
    true_damping = 0.1    # C
    mean_noise = 0.5      # Mean of Gaussian noise
    std_noise = 0.2       # Standard deviation of Gaussian noise

    # Generate Gaussian stochastic forcing
    noise_force = np.random.normal(mean_noise, std_noise, size=len(t))

    # Generate training data
    x0 = [0, 0]
    X = np.zeros((len(t), 2))
    X[0] = x0

    for i in range(len(t) - 1):
        sol = odeint(structural_dynamics, X[i], [t[i], t[i + 1]], args=(noise_force[i],))
        X[i + 1] = sol[-1]

    # Calculate derivatives using finite differences
    X_dot = np.gradient(X, dt, axis=0)

    # Fit Bayesian SINDy model
    model = BayesianSINDy(n_params=8, b=0.01)
    model.fit(X, X_dot, n_steps=10000)

    # Generate predictions with uncertainty for training data
    predictions = model.predict(x0, t)

    # Plot true vs predicted states for training data
    plot_true_vs_predicted(t, X, predictions)

    # Plot posterior parameter distributions
    plot_parameter_distributions(model, model.feature_names)

    # Compute true coefficients
    true_coeffs = [0, true_stiffness / true_mass, true_damping / true_mass, 0, 0, 0, 0, 0]  # Includes placeholders for unused coefficients

    # Print parameter comparison
    print_parameter_comparison(true_coeffs, model, model.feature_names)

    # Robustness test: Different initial conditions
    print("\n### Robustness Test: Different Initial Conditions ###")
    new_x0 = [1.0, -1.0]  # New initial conditions
    X_new = np.zeros((len(t), 2))
    X_new[0] = new_x0

    for i in range(len(t) - 1):
        sol = odeint(structural_dynamics, X_new[i], [t[i], t[i + 1]], args=(noise_force[i],))
        X_new[i + 1] = sol[-1]

    # Predict for new initial conditions
    predictions_new = model.predict(new_x0, t)

    # Plot true vs predicted states for new initial conditions
    plot_true_vs_predicted(t, X_new, predictions_new)

    # Compute RMSE for robustness evaluation
    rmse_x = np.sqrt(np.mean((X_new[:, 0] - np.mean(predictions_new[:, :, 0], axis=0))**2))
    rmse_v = np.sqrt(np.mean((X_new[:, 1] - np.mean(predictions_new[:, :, 1], axis=0))**2))

    print("\n### RMSE for New Initial Conditions ###")
    print(f"x RMSE: {rmse_x:.5f}")
    print(f"v RMSE: {rmse_v:.5f}")
        
    

    



