import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import emcee
import corner
from tqdm import tqdm

class BayesianSINDy:
    def __init__(self, n_params=1, n_walkers=32, b=0.01):
        """
        Initialize the Bayesian SINDy model.

        Parameters:
            n_params (int): Total number of parameters to estimate (only sigma_epsilon).
            n_walkers (int): Number of walkers for MCMC.
            b (float): Scale parameter for the Laplace prior (sparsity).
        """
        self.n_params = n_params
        self.n_walkers = n_walkers
        self.samples = None
        self.feature_names = ['Sigma_epsilon']
        self.b = b  # Sparsity parameter

    def log_prior(self, theta):
        """
        Compute the log prior probability of the parameters.

        Parameters:
            theta (ndarray): Parameter vector (only sigma_epsilon).

        Returns:
            float: Log prior probability.
        """
        sigma_epsilon = theta[0]

        # Physical constraint: sigma_epsilon must be positive
        if sigma_epsilon <= 0:
            return -np.inf  # Outside physical bounds

        # Half-Cauchy prior for sigma_epsilon
        scale_sigma_epsilon = 1.0
        log_prior = -np.log(np.pi * scale_sigma_epsilon * (1 + (sigma_epsilon / scale_sigma_epsilon) ** 2))

        return log_prior

    def log_likelihood(self, theta, X, X_dot, theta_0, K, C):
        """
        Compute the log likelihood of the data given the parameters.

        Parameters:
            theta (ndarray): Parameter vector (only sigma_epsilon).
            X (ndarray): State matrix with shape (n_samples, 2).
            X_dot (ndarray): Derivatives of the state variables with shape (n_samples, 2).
            theta_0 (float): Fixed mean force.
            K (float): Fixed stiffness.
            C (float): Fixed damping.

        Returns:
            float: Log likelihood.
        """
        try:
            sigma_epsilon = theta[0]

            # Predicted derivatives from the model
            x_dot_pred = X[:, 1]  # dx/dt = v
            v_dot_pred = theta_0 - K * X[:, 0] - C * X[:, 1]

            # Measurement noise standard deviations (assumed negligible)
            sigma_x = 1e-8  # Position derivative noise treated as exact

            # Compute log-likelihood for position and velocity derivatives
            log_like_x = -0.5 * np.sum(((X_dot[:, 0] - x_dot_pred) / sigma_x) ** 2 + np.log(2 * np.pi * sigma_x ** 2))
            log_like_v = -0.5 * np.sum(((X_dot[:, 1] - v_dot_pred) / sigma_epsilon) ** 2 + np.log(2 * np.pi * sigma_epsilon ** 2))

            return log_like_x + log_like_v

        except:
            return -np.inf

    def log_probability(self, theta, X, X_dot, theta_0, K, C):
        """
        Compute the log posterior probability of the parameters.

        Parameters:
            theta (ndarray): Parameter vector (only sigma_epsilon).
            X (ndarray): State matrix with shape (n_samples, 2).
            X_dot (ndarray): Derivatives of the state variables with shape (n_samples, 2).
            theta_0 (float): Fixed mean force.
            K (float): Fixed stiffness.
            C (float): Fixed damping.

        Returns:
            float: Log posterior probability.
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta, X, X_dot, theta_0, K, C)
        return lp + ll

    def fit(self, X, X_dot, theta_0, K, C, n_steps=5000):
        """
        Perform MCMC sampling to estimate sigma_epsilon.

        Parameters:
            X (ndarray): State matrix with shape (n_samples, 2).
            X_dot (ndarray): Derivatives of the state variables with shape (n_samples, 2).
            theta_0 (float): Fixed mean force.
            K (float): Fixed stiffness.
            C (float): Fixed damping.
            n_steps (int): Number of MCMC steps.

        Returns:
            self: Fitted model.
        """
        # Initial guess for sigma_epsilon
        initial_guess = np.array([0.1])  # Starting value for sigma_epsilon

        # Initialize walkers around the initial guess
        pos = initial_guess + 0.01 * np.random.randn(self.n_walkers, self.n_params)
        pos = np.abs(pos)  # Ensure sigma_epsilon is positive

        # Initialize and run the sampler
        sampler = emcee.EnsembleSampler(
            self.n_walkers, self.n_params, self.log_probability,
            args=(X, X_dot, theta_0, K, C)
        )

        print("Running MCMC...")
        # Burn-in phase
        sampler.run_mcmc(pos, 1000, progress=True)
        sampler.reset()
        # Sampling phase
        sampler.run_mcmc(None, n_steps, progress=True)

        # Thinning and flattening the chain
        self.samples = sampler.get_chain(discard=100, thin=10, flat=True)

        return self

    def predict(self, x0, t, theta_0, K, C, n_samples=100):
        """
        Generate predictions with uncertainty by sampling from posterior of sigma_epsilon.

        Parameters:
            x0 (list or ndarray): Initial state [x, v].
            t (ndarray): Time array.
            theta_0 (float): Fixed mean force.
            K (float): Fixed stiffness.
            C (float): Fixed damping.
            n_samples (int): Number of posterior samples to use for prediction.

        Returns:
            ndarray: Array of predictions with shape (n_samples, len(t), 2).
        """
        predictions = []
        idx = np.random.randint(len(self.samples), size=n_samples)
        sigma_epsilon_samples = self.samples[idx, 0]  # Extract sigma_epsilon samples

        for sigma_epsilon in tqdm(sigma_epsilon_samples, desc="Generating predictions"):
            X_pred = np.zeros((len(t), 2))
            X_pred[0] = x0

            # Generate new noise sequence for each prediction
            noise_force = np.random.normal(0, sigma_epsilon, size=len(t))

            for i in range(len(t) - 1):
                def system(state, t_val):
                    x, v = state
                    dx = v
                    dv = theta_0 - K * x - C * v + noise_force[i]
                    return [dx, dv]

                sol = odeint(system, X_pred[i], [t[i], t[i + 1]])
                X_pred[i + 1] = sol[-1]

            predictions.append(X_pred)

        return np.array(predictions)

def plot_true_vs_predicted(t, X_true, predictions, state_labels=['Position', 'Velocity']):
    """
    Plot true vs. predicted states with confidence intervals.

    Parameters:
        t (ndarray): Time array.
        X_true (ndarray): True state matrix with shape (n_samples, 2).
        predictions (ndarray): Predicted states with shape (n_samples, len(t), 2).
        state_labels (list): Labels for the state variables.
    """
    # Calculate prediction statistics
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    for i in range(2):
        axs[i].plot(t, X_true[:, i], 'b-', label=f'True {state_labels[i]}')
        axs[i].plot(t, pred_mean[:, i], 'r--', label=f'Mean Prediction {state_labels[i]}')
        axs[i].fill_between(t,
                            pred_mean[:, i] - 2 * pred_std[:, i],
                            pred_mean[:, i] + 2 * pred_std[:, i],
                            color='r', alpha=0.2, label='95% CI')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel(state_labels[i])
        axs[i].legend()
        axs[i].set_title(f'{state_labels[i]} Response with Uncertainty')
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

def plot_parameter_distributions(model, param_names=['Sigma_epsilon']):
    """
    Plot posterior distributions of parameters using the corner library.

    Parameters:
        model (BayesianSINDy): Fitted Bayesian SINDy model.
        param_names (list): Names of the parameters.
    """
    fig = corner.corner(model.samples,
                        labels=param_names,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12})
    plt.show()

def print_parameter_comparison(true_values, model, param_names=['Sigma_epsilon']):
    """
    Compare true coefficients with estimated parameters and report errors.

    Parameters:
        true_values (list): True values of the parameters.
        model (BayesianSINDy): Fitted Bayesian SINDy model.
        param_names (list): Names of the parameters.
    """
    print("\n### Parameter Comparison ###\n")

    def format_comparison(true_val, estimates, label):
        median = np.median(estimates)
        std = np.std(estimates)
        if true_val == 0:
            # Use absolute error for parameters with true value of 0
            absolute_error = np.abs(median)
            return f"{label:20}: True = {true_val:.5f}, Estimate = {median:.5f} ± {std:.5f}, Abs. Error = {absolute_error:.5f}"
        else:
            # Use percentage error for nonzero parameters
            percentage_error = 100 * np.abs(median - true_val) / np.abs(true_val)
            return f"{label:20}: True = {true_val:.5f}, Estimate = {median:.5f} ± {std:.5f}, Error = {percentage_error:.2f}%"

    for i, name in enumerate(param_names):
        print(format_comparison(true_values[i], model.samples[:, i], name))

# Define the true system dynamics for generating training data
def structural_dynamics(state, t, noise_force):
    """
    Defines the structural dynamics of the 1DOF system.

    Parameters:
        state (list or ndarray): Current state [x, v].
        t (float): Current time.
        noise_force (float): Force acting on the mass at time t.

    Returns:
        list: Derivatives [dx/dt, dv/dt].
    """
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
    theta_0 = 0.5         # Mean force
    std_noise = 0.1       # Standard deviation of process noise (sigma_epsilon)

    # Generate Gaussian stochastic forcing (process noise only)
    noise_force = np.random.normal(theta_0, std_noise, size=len(t))

    # Generate training data with exact derivatives
    x0 = [0, 0]
    X = np.zeros((len(t), 2))
    X[0] = x0
    X_dot_exact = np.zeros((len(t), 2))

    for i in range(len(t) - 1):
        sol = odeint(structural_dynamics, X[i], [t[i], t[i + 1]], args=(noise_force[i],))
        X[i + 1] = sol[-1]
        # Compute exact derivatives using the ODE function
        X_dot_exact[i] = structural_dynamics(X[i], t[i], noise_force[i])

    # For the last point
    X_dot_exact[-1] = structural_dynamics(X[-1], t[-1], noise_force[-1])

    # Compute derivatives using finite differences
    X_dot_fd = np.gradient(X, dt, axis=0)

    # Fit Bayesian SINDy model using finite difference derivatives
    print("\nFitting model using finite difference derivatives...")
    model_fd = BayesianSINDy(n_params=1, n_walkers=32, b=0.01)
    model_fd.fit(X, X_dot_fd, theta_0, true_stiffness / true_mass, true_damping / true_mass, n_steps=2000)

    # Fit Bayesian SINDy model using exact derivatives
    print("\nFitting model using exact derivatives...")
    model_exact = BayesianSINDy(n_params=1, n_walkers=32, b=0.01)
    model_exact.fit(X, X_dot_exact, theta_0, true_stiffness / true_mass, true_damping / true_mass, n_steps=2000)

    # Compare estimated sigma_epsilon
    sigma_epsilon_fd = np.median(model_fd.samples[:, 0])
    sigma_epsilon_exact = np.median(model_exact.samples[:, 0])

    print("\n### Estimated sigma_epsilon ###")
    print(f"Using Finite Difference Derivatives: {sigma_epsilon_fd:.5f}")
    print(f"Using Exact Derivatives:           {sigma_epsilon_exact:.5f}")
    print(f"True sigma_epsilon:               {std_noise:.5f}")

    # Analyze errors in derivative estimation
    error_x_dot = X_dot_fd[:, 0] - X_dot_exact[:, 0]
    error_v_dot = X_dot_fd[:, 1] - X_dot_exact[:, 1]

    print("\n### Derivative Estimation Errors ###")
    print(f"Mean absolute error in x_dot (FD vs. Exact): {np.mean(np.abs(error_x_dot)):.5e}")
    print(f"Mean absolute error in v_dot (FD vs. Exact): {np.mean(np.abs(error_v_dot)):.5e}")

    # Visualization for Finite Difference Model
    print("\n### Visualization for Finite Difference Model ###")
    predictions_fd = model_fd.predict(x0, t, theta_0, true_stiffness / true_mass, true_damping / true_mass, n_samples=200)
    plot_true_vs_predicted(t, X, predictions_fd, state_labels=['Position', 'Velocity'])
    plot_parameter_distributions(model_fd, param_names=['Sigma_epsilon (FD)'])
    print_parameter_comparison([std_noise], model_fd, param_names=['Sigma_epsilon (FD)'])

    # Visualization for Exact Derivatives Model
    print("\n### Visualization for Exact Derivatives Model ###")
    predictions_exact = model_exact.predict(x0, t, theta_0, true_stiffness / true_mass, true_damping / true_mass, n_samples=200)
    plot_true_vs_predicted(t, X, predictions_exact, state_labels=['Position', 'Velocity'])
    plot_parameter_distributions(model_exact, param_names=['Sigma_epsilon (Exact)'])
    print_parameter_comparison([std_noise], model_exact, param_names=['Sigma_epsilon (Exact)'])

    # Robustness test: Different initial conditions
    print("\n### Robustness Test: Different Initial Conditions ###")
    new_x0 = [1.0, -1.0]  # New initial conditions [x, v]
    X_new = np.zeros((len(t), 2))
    X_new[0] = new_x0

    for i in range(len(t) - 1):
        sol = odeint(structural_dynamics, X_new[i], [t[i], t[i + 1]], args=(noise_force[i],))
        X_new[i + 1] = sol[-1]
        # Compute exact derivatives using the ODE function
        X_dot_exact_new = structural_dynamics(X_new[i], t[i], noise_force[i])



    # Compute derivatives using finite differences for new initial conditions
    X_dot_fd_new = np.gradient(X_new, dt, axis=0)

    # Compute exact derivatives for new initial conditions
    X_dot_exact_new = np.zeros((len(t), 2))
    for i in range(len(t)):
        X_dot_exact_new[i] = structural_dynamics(X_new[i], t[i], noise_force[i])

    # Fit Bayesian SINDy model using finite difference derivatives for new initial conditions
    print("\nFitting model for new initial conditions using finite difference derivatives...")
    model_fd_new = BayesianSINDy(n_params=1, n_walkers=32, b=0.01)
    model_fd_new.fit(X_new, X_dot_fd_new, theta_0, true_stiffness / true_mass, true_damping / true_mass, n_steps=5000)

    # Fit Bayesian SINDy model using exact derivatives for new initial conditions
    print("\nFitting model for new initial conditions using exact derivatives...")
    model_exact_new = BayesianSINDy(n_params=1, n_walkers=32, b=0.01)
    model_exact_new.fit(X_new, X_dot_exact_new, theta_0, true_stiffness / true_mass, true_damping / true_mass, n_steps=5000)

    # Analyze errors in derivative estimation for new initial conditions
    error_x_dot_new = np.zeros(len(t))
    error_v_dot_new = np.zeros(len(t))

    for i in range(len(t)):
        # Compute exact derivatives using the ODE function
        X_dot_exact_new_i = structural_dynamics(X_new[i], t[i], noise_force[i])
        # Compute errors between finite difference and exact derivatives
        error_x_dot_new[i] = X_dot_fd_new[i, 0] - X_dot_exact_new_i[0]
        error_v_dot_new[i] = X_dot_fd_new[i, 1] - X_dot_exact_new_i[1]

    print("\n### Derivative Estimation Errors for New Initial Conditions ###")
    print(f"Mean absolute error in x_dot (FD vs. Exact): {np.mean(np.abs(error_x_dot_new)):.5e}")
    print(f"Mean absolute error in v_dot (FD vs. Exact): {np.mean(np.abs(error_v_dot_new)):.5e}")



    # Visualization for Finite Difference Model with New Initial Conditions
    print("\n### Visualization for Finite Difference Model with New Initial Conditions ###")
    predictions_fd_new = model_fd_new.predict(new_x0, t, theta_0, true_stiffness / true_mass, true_damping / true_mass, n_samples=200)
    plot_true_vs_predicted(t, X_new, predictions_fd_new, state_labels=['Position', 'Velocity'])
    plot_parameter_distributions(model_fd_new, param_names=['Sigma_epsilon (FD - New)'])
    print_parameter_comparison([std_noise], model_fd_new, param_names=['Sigma_epsilon (FD - New)'])

    # Visualization for Exact Derivatives Model with New Initial Conditions
    print("\n### Visualization for Exact Derivatives Model with New Initial Conditions ###")
    predictions_exact_new = model_exact_new.predict(new_x0, t, theta_0, true_stiffness / true_mass, true_damping / true_mass, n_samples=200)
    plot_true_vs_predicted(t, X_new, predictions_exact_new, state_labels=['Position', 'Velocity'])
    plot_parameter_distributions(model_exact_new, param_names=['Sigma_epsilon (Exact - New)'])
    print_parameter_comparison([std_noise], model_exact_new, param_names=['Sigma_epsilon (Exact - New)'])

    # Compute RMSE for original initial conditions
    rmse_x_fd = np.sqrt(np.mean((X[:, 0] - np.mean(predictions_fd[:, :, 0], axis=0)) ** 2))
    rmse_v_fd = np.sqrt(np.mean((X[:, 1] - np.mean(predictions_fd[:, :, 1], axis=0)) ** 2))
    rmse_x_exact = np.sqrt(np.mean((X[:, 0] - np.mean(predictions_exact[:, :, 0], axis=0)) ** 2))
    rmse_v_exact = np.sqrt(np.mean((X[:, 1] - np.mean(predictions_exact[:, :, 1], axis=0)) ** 2))

    print("\n### RMSE for Original Initial Conditions ###")
    print(f"Finite Difference - x RMSE: {rmse_x_fd:.5f}")
    print(f"Finite Difference - v RMSE: {rmse_v_fd:.5f}")
    print(f"Exact Derivatives  - x RMSE: {rmse_x_exact:.5f}")
    print(f"Exact Derivatives  - v RMSE: {rmse_v_exact:.5f}")

    # Compute RMSE for new initial conditions
    rmse_x_fd_new = np.sqrt(np.mean((X_new[:, 0] - np.mean(predictions_fd_new[:, :, 0], axis=0)) ** 2))
    rmse_v_fd_new = np.sqrt(np.mean((X_new[:, 1] - np.mean(predictions_fd_new[:, :, 1], axis=0)) ** 2))
    rmse_x_exact_new = np.sqrt(np.mean((X_new[:, 0] - np.mean(predictions_exact_new[:, :, 0], axis=0)) ** 2))
    rmse_v_exact_new = np.sqrt(np.mean((X_new[:, 1] - np.mean(predictions_exact_new[:, :, 1], axis=0)) ** 2))

    print("\n### RMSE for New Initial Conditions ###")
    print(f"Finite Difference - x RMSE: {rmse_x_fd_new:.5f}")
    print(f"Finite Difference - v RMSE: {rmse_v_fd_new:.5f}")
    print(f"Exact Derivatives  - x RMSE: {rmse_x_exact_new:.5f}")
    print(f"Exact Derivatives  - v RMSE: {rmse_v_exact_new:.5f}")
