import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import emcee
import corner
from tqdm import tqdm

class BayesianSINDy:
    def __init__(self, n_params=6, n_walkers=32, b=0.01):
        self.n_params = n_params
        self.n_walkers = n_walkers
        self.samples = None
        self.feature_names = ['Theta_0', 'K1', 'K2', 'C1', 'C2', 'Sigma_epsilon']
        self.b = b  # Sparsity parameter

    def log_prior(self, theta):
        """
        Priors for the parameters.
        """
        theta_0, K1, K2, C1, C2, sigma_epsilon = theta

        # Physical constraints
        if not (0 < K1 < 5.0 and 0 < K2 < 5.0 and 0 < C1 < 1.0 and 0 < C2 < 1.0 and sigma_epsilon > 0):
            return -np.inf  # Outside physical bounds

        # Initialize the log-prior
        log_prior = 0

        # Laplace prior for sparsity (K1, K2, C1, C2)
        for param in [K1, K2, C1, C2]:
            log_prior += -np.log(2 * self.b) - np.abs(param) / self.b

        # Gaussian prior for theta_0 (mean force)
        mu_theta0 = 0.0
        sigma_theta0 = 1.0
        log_prior += -0.5 * ((theta_0 - mu_theta0) / sigma_theta0)**2 - np.log(sigma_theta0 * np.sqrt(2 * np.pi))

        # Prior for sigma_epsilon (process noise)
        scale_sigma_epsilon = 1.0
        log_prior += -np.log(np.pi * scale_sigma_epsilon * (1 + (sigma_epsilon / scale_sigma_epsilon)**2))

        return log_prior

    def log_likelihood(self, theta, X, X_dot):
        """
        Likelihood function accounting for unobserved process noise.
        """
        try:
            theta_0, K1, K2, C1, C2, sigma_epsilon = theta

            # Extract states
            x1 = X[:, 0]
            v1 = X[:, 1]
            x2 = X[:, 2]
            v2 = X[:, 3]

            # Predicted derivatives from the model
            x1_dot_pred = v1
            v1_dot_pred = -C1 * (v1 - v2) - K1 * (x1 - x2)
            x2_dot_pred = v2
            v2_dot_pred = theta_0 - C1 * (v2 - v1) - K1 * (x2 - x1) - C2 * v2 - K2 * x2

            # Measurement noise standard deviations (set to near zero)
            sigma_x = 1e-8  # Position derivative noise treated as exact

            # Total variance for dv/dt (process noise)
            sigma_v1 = 1e-8  # Small process noise for v1_dot
            sigma_v2 = sigma_epsilon  # Process noise for v2_dot

            # Compute log-likelihood for position and velocity derivatives
            log_like_x1 = -0.5 * np.sum(((X_dot[:, 0] - x1_dot_pred) / sigma_x)**2 + np.log(2 * np.pi * sigma_x**2))
            log_like_v1 = -0.5 * np.sum(((X_dot[:, 1] - v1_dot_pred) / sigma_v1)**2 + np.log(2 * np.pi * sigma_v1**2))
            log_like_x2 = -0.5 * np.sum(((X_dot[:, 2] - x2_dot_pred) / sigma_x)**2 + np.log(2 * np.pi * sigma_x**2))
            log_like_v2 = -0.5 * np.sum(((X_dot[:, 3] - v2_dot_pred) / sigma_v2)**2 + np.log(2 * np.pi * sigma_v2**2))

            return log_like_x1 + log_like_v1 + log_like_x2 + log_like_v2

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
        # Initial guess
        initial_guess = np.array([0.5, 1.0, 2.0, 0.1, 0.2, 0.1])  # [theta_0, K1, K2, C1, C2, sigma_epsilon]

        # Initialize walkers around the initial guess
        pos = initial_guess + 0.1 * np.random.randn(self.n_walkers, self.n_params)

        # Ensure parameters are within bounds
        pos[:, 1] = np.abs(pos[:, 1])  # K1 (positive)
        pos[:, 2] = np.abs(pos[:, 2])  # K2 (positive)
        pos[:, 3] = np.abs(pos[:, 3])  # C1 (positive)
        pos[:, 4] = np.abs(pos[:, 4])  # C2 (positive)
        pos[:, 5] = np.abs(pos[:, 5])  # sigma_epsilon (positive)

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
            theta_0, K1, K2, C1, C2, sigma_epsilon = theta
            X_pred = np.zeros((len(t), 4))
            X_pred[0] = x0

            # Generate new noise sequence for each prediction
            noise_force = np.random.normal(theta_0, sigma_epsilon, size=len(t))

            for i in range(len(t)-1):
                def system(state, t_val):
                    x1, v1, x2, v2 = state
                    dx1 = v1
                    dv1 = -C1 * (v1 - v2) - K1 * (x1 - x2)
                    dx2 = v2
                    dv2 = noise_force[i] - C1 * (v2 - v1) - K1 * (x2 - x1) - C2 * v2 - K2 * x2
                    return [dx1, dv1, dx2, dv2]

                sol = odeint(system, X_pred[i], [t[i], t[i+1]])
                X_pred[i+1] = sol[-1]

            predictions.append(X_pred)

        return np.array(predictions)

def plot_true_vs_predicted(t, X_true, predictions):
    """
    Plot true vs. predicted states with confidence intervals.
    """
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)

    fig, axs = plt.subplots(4, 1, figsize=(10, 16))

    # x1
    axs[0].plot(t, X_true[:, 0], 'b-', label='True x1')
    axs[0].plot(t, pred_mean[:, 0], 'r--', label='Mean Prediction x1')
    axs[0].fill_between(t, pred_mean[:, 0] - 2 * pred_std[:, 0], pred_mean[:, 0] + 2 * pred_std[:, 0],
                        color='r', alpha=0.2, label='95% CI')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('x1')
    axs[0].legend()
    axs[0].set_title('x1 with Uncertainty')
    axs[0].grid(True)

    # v1
    axs[1].plot(t, X_true[:, 1], 'b-', label='True v1')
    axs[1].plot(t, pred_mean[:, 1], 'r--', label='Mean Prediction v1')
    axs[1].fill_between(t, pred_mean[:, 1] - 2 * pred_std[:, 1], pred_mean[:, 1] + 2 * pred_std[:, 1],
                        color='r', alpha=0.2, label='95% CI')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('v1')
    axs[1].legend()
    axs[1].set_title('v1 with Uncertainty')
    axs[1].grid(True)

    # x2
    axs[2].plot(t, X_true[:, 2], 'b-', label='True x2')
    axs[2].plot(t, pred_mean[:, 2], 'r--', label='Mean Prediction x2')
    axs[2].fill_between(t, pred_mean[:, 2] - 2 * pred_std[:, 2], pred_mean[:, 2] + 2 * pred_std[:, 2],
                        color='r', alpha=0.2, label='95% CI')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('x2')
    axs[2].legend()
    axs[2].set_title('x2 with Uncertainty')
    axs[2].grid(True)

    # v2
    axs[3].plot(t, X_true[:, 3], 'b-', label='True v2')
    axs[3].plot(t, pred_mean[:, 3], 'r--', label='Mean Prediction v2')
    axs[3].fill_between(t, pred_mean[:, 3] - 2 * pred_std[:, 3], pred_mean[:, 3] + 2 * pred_std[:, 3],
                        color='r', alpha=0.2, label='95% CI')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('v2')
    axs[3].legend()
    axs[3].set_title('v2 with Uncertainty')
    axs[3].grid(True)

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
    """
    Simulates the 2DoF structural dynamics system with stochastic force applied to mass 2.
    """
    # System parameters
    m1 = 1.0
    m2 = 1.0
    k1 = 4.0
    k2 = 1.0
    c1 = 0.2
    c2 = 0.2

    x1, v1, x2, v2 = state

    dx1 = v1
    dv1 = (-c1 * (v1 - v2) - k1 * (x1 - x2)) / m1
    dx2 = v2
    dv2 = (noise_force - c1 * (v2 - v1) - k1 * (x2 - x1) - c2 * v2 - k2 * x2) / m2

    return [dx1, dv1, dx2, dv2]

# Main execution
if __name__ == "__main__":
    # System parameters and data generation
    dt = 0.005
    t = np.arange(0, 20, dt)
    np.random.seed(42)

    # True model parameters
    m1 = 1.0
    m2 = 1.0
    k1 = 4.0     # K1
    k2 = 1.0     # K2
    c1 = 0.2     # C1
    c2 = 0.2     # C2
    theta_0 = 0.5  # Mean force on mass 2
    sigma_epsilon = 0.05  # Standard deviation of Gaussian noise (process noise)

    # Generate Gaussian stochastic forcing (process noise only)
    noise_force = np.random.normal(theta_0, sigma_epsilon, size=len(t))

    # Generate training data
    x0 = [0.0, 0.0, 0.0, 0.0]  # [x1_0, v1_0, x2_0, v2_0]
    X = np.zeros((len(t), 4))
    X[0] = x0
    X_dot = np.zeros((len(t), 4))

    for i in range(len(t) - 1):
        sol = odeint(structural_dynamics, X[i], [t[i], t[i + 1]], args=(noise_force[i],))
        X[i + 1] = sol[-1]
        X_dot[i] = structural_dynamics(X[i], t[i], noise_force[i])

    # For the last point
    X_dot[-1] = structural_dynamics(X[-1], t[-1], noise_force[-1])

    # Fit Bayesian SINDy model (no measurement noise)
    model = BayesianSINDy(n_params=6, n_walkers=64, b=0.1)
    model.fit(X, X_dot, n_steps=2000)

    # Generate predictions with uncertainty for training data
    predictions = model.predict(x0, t)

    # Plot true vs predicted states for training data
    plot_true_vs_predicted(t, X, predictions)

    # Plot posterior parameter distributions
    plot_parameter_distributions(model, model.feature_names)

    # Compute true coefficients
    # Parameters: [theta_0, K1, K2, C1, C2, sigma_epsilon]
    true_coeffs = [theta_0, k1 / m1, k2 / m2, c1 / m1, c2 / m2, sigma_epsilon]

    # Print parameter comparison
    print_parameter_comparison(true_coeffs, model, model.feature_names)

    # Robustness test: Different initial conditions
    print("\n### Robustness Test: Different Initial Conditions ###")
    new_x0 = [1.0, -1.0, 0.5, 0.5]  # New initial conditions
    X_new = np.zeros((len(t), 4))
    X_new[0] = new_x0

    for i in range(len(t) - 1):
        sol = odeint(structural_dynamics, X_new[i], [t[i], t[i + 1]], args=(noise_force[i],))
        X_new[i + 1] = sol[-1]

    # Compute derivatives for new initial conditions
    X_new_dot = np.zeros_like(X_new)
    for i in range(len(t) - 1):
        X_new_dot[i] = structural_dynamics(X_new[i], t[i], noise_force[i])
    X_new_dot[-1] = structural_dynamics(X_new[-1], t[-1], noise_force[-1])

    # Predict for new initial conditions
    predictions_new = model.predict(new_x0, t)

    # Plot true vs predicted states for new initial conditions
    plot_true_vs_predicted(t, X_new, predictions_new)

    # Compute RMSE for robustness evaluation
    rmse_x1 = np.sqrt(np.mean((X_new[:, 0] - np.mean(predictions_new[:, :, 0], axis=0)) ** 2))
    rmse_v1 = np.sqrt(np.mean((X_new[:, 1] - np.mean(predictions_new[:, :, 1], axis=0)) ** 2))
    rmse_x2 = np.sqrt(np.mean((X_new[:, 2] - np.mean(predictions_new[:, :, 2], axis=0)) ** 2))
    rmse_v2 = np.sqrt(np.mean((X_new[:, 3] - np.mean(predictions_new[:, :, 3], axis=0)) ** 2))

    print("\n### RMSE for New Initial Conditions ###")
    print(f"x1 RMSE: {rmse_x1:.5f}")
    print(f"v1 RMSE: {rmse_v1:.5f}")
    print(f"x2 RMSE: {rmse_x2:.5f}")
    print(f"v2 RMSE: {rmse_v2:.5f}")
