import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import emcee
import corner
from tqdm import tqdm

class BayesianSINDy:
    def __init__(self, n_walkers=32, b=0.01):
        self.n_params = None  # To be set after constructing the library
        self.n_features = None  # Number of features in the library
        self.n_walkers = n_walkers
        self.samples = None
        self.b = b  # Scale parameter for sparsity
        self.feature_names = None  # To be set after constructing the library


    def construct_library(self, X):
        """
        Construct the library of candidate functions (Theta matrix), including selected quadratic terms.
        """
        x1 = X[:, 0]
        v1 = X[:, 1]
        x2 = X[:, 2]
        v2 = X[:, 3]
    
        variables = [x1, v1, x2, v2]
        var_names = ['x1', 'v1', 'x2', 'v2']
    
        Theta = [np.ones_like(x1)]
        self.feature_names = ['1']
    
        # Add linear terms
        for var, name in zip(variables, var_names):
            Theta.append(var)
            self.feature_names.append(name)
    
        # Add selected quadratic terms
        quadratic_terms = [(x1, 'x1^2'), (v1, 'v1^2'), (x2, 'x2^2'), (v2, 'v2^2')]
        for var, name in quadratic_terms:
            Theta.append(var ** 2)
            self.feature_names.append(name)
    
        Theta = np.column_stack(Theta)
    
        return Theta



    def log_prior(self, theta):
        """
        Priors for the parameters with adjustable sparsity via the Laplace scale parameter b.
        """
        # Unpack parameters
        coeffs_f1 = theta[:self.n_features]  # Coefficients for dv1/dt
        coeffs_f2 = theta[self.n_features:2 * self.n_features]  # Coefficients for dv2/dt
        sigma_epsilon = theta[-1]  # Process noise

        # Physical constraints: Ensure sigma_epsilon is positive
        if sigma_epsilon <= 0:
            return -np.inf  # Outside physical bounds

        # Initialize the log-prior
        log_prior = 0

        # Laplace prior for sparsity on coefficients_f1 and coefficients_f2
        for param in np.concatenate([coeffs_f1, coeffs_f2]):
            log_prior += -np.log(2 * self.b) - np.abs(param) / self.b

        # Half-Cauchy prior for sigma_epsilon (process noise)
        scale_sigma_epsilon = 1.0
        log_prior += -np.log(np.pi * scale_sigma_epsilon * (1 + (sigma_epsilon / scale_sigma_epsilon) ** 2))

        return log_prior

    def log_likelihood(self, theta, Theta, X_dot):
        """
        Likelihood function accounting for unobserved process noise.
        """
        try:
            # Unpack parameters
            coeffs_f1 = theta[:self.n_features]  # Coefficients for dv1/dt
            coeffs_f2 = theta[self.n_features:2 * self.n_features]  # Coefficients for dv2/dt
            sigma_epsilon = theta[-1]  # Process noise

            # Predicted derivatives from the model
            x1_dot_pred = X_dot[:, 1]  # dx1/dt = v1 (exact, no measurement noise)
            x2_dot_pred = X_dot[:, 3]  # dx2/dt = v2 (exact, no measurement noise)

            # Predicted accelerations using the inferred model
            v1_dot_pred = Theta @ coeffs_f1
            v2_dot_pred = Theta @ coeffs_f2

            # Measurement noise standard deviations (set to near zero to treat as exact)
            sigma_x = 1e-8  # Effectively no measurement noise for position derivatives

            # Total variance for dv/dt (only process noise)
            sigma_total_v = sigma_epsilon

            # Compute log-likelihood for position derivatives (dx/dt)
            log_like = 0
            log_like += -0.5 * np.sum(((X_dot[:, 0] - x1_dot_pred) / sigma_x) ** 2 + np.log(2 * np.pi * sigma_x ** 2))
            log_like += -0.5 * np.sum(((X_dot[:, 2] - x2_dot_pred) / sigma_x) ** 2 + np.log(2 * np.pi * sigma_x ** 2))

            # Compute log-likelihood for accelerations (dv/dt)
            log_like += -0.5 * np.sum(((X_dot[:, 1] - v1_dot_pred) / sigma_total_v) ** 2 + np.log(2 * np.pi * sigma_total_v ** 2))
            log_like += -0.5 * np.sum(((X_dot[:, 3] - v2_dot_pred) / sigma_total_v) ** 2 + np.log(2 * np.pi * sigma_total_v ** 2))

            return log_like

        except:
            return -np.inf

    def log_probability(self, theta, Theta, X_dot):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta, Theta, X_dot)
        return lp + ll
    
    def fit(self, X, X_dot, n_steps=5000):
        """
        Perform MCMC sampling.
        """
        # Construct the library of candidate functions
        Theta = self.construct_library(X)
        self.n_features = Theta.shape[1]
        # Number of parameters: coefficients_f1 + coefficients_f2 + sigma_epsilon
        self.n_params = 2 * self.n_features + 1
    
        # Set the number of walkers (at least 2 * n_params)
        self.n_walkers = max(2 * self.n_params, self.n_walkers)
    
        # Initial guess for parameters
        initial_guess = np.concatenate([
            np.zeros(self.n_features),  # coeffs_f1
            np.zeros(self.n_features),  # coeffs_f2
            np.array([1.0])  # sigma_epsilon
        ])
    
        # Use a smaller spread for initial positions
        pos = initial_guess + 0.05 * np.random.randn(self.n_walkers, self.n_params)
    
        # Ensure sigma_epsilon is positive and not too small
        pos[:, -1] = np.abs(pos[:, -1]) + 0.1  # sigma_epsilon (positive)
    
        # Initialize and run the sampler
        sampler = emcee.EnsembleSampler(
            self.n_walkers, self.n_params, self.log_probability,
            args=(Theta, X_dot)
        )
    
        print("Running MCMC...")
        # Burn-in phase
        sampler.run_mcmc(pos, 2000, progress=True)
        sampler.reset()
        # Sampling phase
        sampler.run_mcmc(None, n_steps, progress=True)
    
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
    
        dt = t[1] - t[0]  # Assuming uniform time steps
    
        for theta in tqdm(theta_samples, desc="Generating predictions"):
            # Unpack parameters
            coeffs_f1 = theta[:self.n_features]
            coeffs_f2 = theta[self.n_features:2 * self.n_features]
            sigma_epsilon = theta[-1]
    
            X_pred = np.zeros((len(t), 4))
            X_pred[0] = x0
    
            # Generate new noise sequence for each prediction
            process_noise = np.random.normal(0, sigma_epsilon, size=(len(t), 2))  # For dv1/dt and dv2/dt
    
            for i in range(len(t) - 1):
                x1, v1, x2, v2 = X_pred[i]
    
                # Define a function to compute derivatives
                def derivatives(state):
                    x1_, v1_, x2_, v2_ = state
    
                    # Construct Theta_current for current state
                    Theta_current = [1]  # Constant term
                    variables = [x1_, v1_, x2_, v2_]
                    Theta_current.extend(variables)
                    Theta_current.append(x1_ ** 2)
                    Theta_current.append(v1_ ** 2)
                    Theta_current.append(x2_ ** 2)
                    Theta_current.append(v2_ ** 2)
                    Theta_current = np.array(Theta_current)
    
                    # Compute accelerations using inferred model
                    v1_dot = Theta_current @ coeffs_f1
                    v2_dot = Theta_current @ coeffs_f2
    
                    return np.array([v1_, v1_dot, v2_, v2_dot])
    
                # RK4 Integration
                k1 = derivatives(X_pred[i]) + np.array([0, process_noise[i, 0], 0, process_noise[i, 1]])
                k2 = derivatives(X_pred[i] + 0.5 * dt * k1) + np.array([0, process_noise[i, 0], 0, process_noise[i, 1]])
                k3 = derivatives(X_pred[i] + 0.5 * dt * k2) + np.array([0, process_noise[i, 0], 0, process_noise[i, 1]])
                k4 = derivatives(X_pred[i] + dt * k3) + np.array([0, process_noise[i, 0], 0, process_noise[i, 1]])
    
                X_pred[i + 1] = X_pred[i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
            predictions.append(X_pred)
    
        return np.array(predictions)


    def plot_true_vs_predicted(self, t, X_true, predictions, t_max=None):
        """
        Plot true vs. predicted states with confidence intervals up to specified time.
        """
        if t_max is not None:
            # Find the index corresponding to t_max
            idx = np.searchsorted(t, t_max, side='right')
            t = t[:idx]
            X_true = X_true[:idx]
            predictions = predictions[:, :idx]
        else:
            idx = len(t)
    
        # Calculate prediction statistics
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
    
        fig, axs = plt.subplots(4, 1, figsize=(12, 16))

        # Position of mass 1
        axs[0].plot(t, X_true[:, 0], 'b-', label='True Position x1')
        axs[0].plot(t, pred_mean[:, 0], 'r--', label='Mean Prediction x1')
        axs[0].fill_between(t,
                            pred_mean[:, 0] - 2 * pred_std[:, 0],
                            pred_mean[:, 0] + 2 * pred_std[:, 0],
                            color='r', alpha=0.2, label='95% CI')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Position x1')
        axs[0].legend()
        axs[0].set_title('Position x1 Response with Uncertainty')
        axs[0].grid(True)

        # Velocity of mass 1
        axs[1].plot(t, X_true[:, 1], 'b-', label='True Velocity v1')
        axs[1].plot(t, pred_mean[:, 1], 'r--', label='Mean Prediction v1')
        axs[1].fill_between(t,
                            pred_mean[:, 1] - 2 * pred_std[:, 1],
                            pred_mean[:, 1] + 2 * pred_std[:, 1],
                            color='r', alpha=0.2, label='95% CI')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Velocity v1')
        axs[1].legend()
        axs[1].set_title('Velocity v1 Response with Uncertainty')
        axs[1].grid(True)

        # Position of mass 2
        axs[2].plot(t, X_true[:, 2], 'b-', label='True Position x2')
        axs[2].plot(t, pred_mean[:, 2], 'r--', label='Mean Prediction x2')
        axs[2].fill_between(t,
                            pred_mean[:, 2] - 2 * pred_std[:, 2],
                            pred_mean[:, 2] + 2 * pred_std[:, 2],
                            color='r', alpha=0.2, label='95% CI')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Position x2')
        axs[2].legend()
        axs[2].set_title('Position x2 Response with Uncertainty')
        axs[2].grid(True)

        # Velocity of mass 2
        axs[3].plot(t, X_true[:, 3], 'b-', label='True Velocity v2')
        axs[3].plot(t, pred_mean[:, 3], 'r--', label='Mean Prediction v2')
        axs[3].fill_between(t,
                            pred_mean[:, 3] - 2 * pred_std[:, 3],
                            pred_mean[:, 3] + 2 * pred_std[:, 3],
                            color='r', alpha=0.2, label='95% CI')
        axs[3].set_xlabel('Time')
        axs[3].set_ylabel('Velocity v2')
        axs[3].legend()
        axs[3].set_title('Velocity v2 Response with Uncertainty')
        axs[3].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_parameter_distributions(self):
        """
        Plot posterior distributions of parameters.
        """
        # Plot parameter distributions using the corner library
        labels = [f'f1_{name}' for name in self.feature_names] + \
                 [f'f2_{name}' for name in self.feature_names] + \
                 ['Sigma_epsilon']

        fig = corner.corner(self.samples,
                            labels=labels,
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True,
                            title_kwargs={"fontsize": 12})
        plt.show()

def compute_true_coefficients(m1, m2, k1, k2, c1, c2, constant_force, feature_names):
    """
    Computes the true coefficients for the 2DOF system based on the governing equations.
    Parameters:
        m1, m2: Masses of the two bodies.
        k1, k2: Stiffnesses of the springs.
        c1, c2: Damping coefficients.
        constant_force: Constant force applied to mass 2.
        feature_names: List of feature names (from the model)
    Returns:
        true_coeffs_f1: True coefficients for dv1/dt.
        true_coeffs_f2: True coefficients for dv2/dt.
    """
    # Compute true coefficients mapping feature names to values

    # Coefficients for dv1/dt
    coeffs_f1 = {
        '1': 0,
        'x1': -(k1 + k2) / m1,
        'v1': -(c1 + c2) / m1,
        'x2': k2 / m1,
        'v2': c2 / m1
    }

    # Coefficients for dv2/dt
    coeffs_f2 = {
        '1': constant_force / m2,
        'x1': k2 / m2,
        'v1': c2 / m2,
        'x2': -k2 / m2,
        'v2': -c2 / m2
    }

    # For quadratic terms, coefficients are zero
    quadratic_terms = [name for name in feature_names if '*' in name or '^' in name]
    for term in quadratic_terms:
        coeffs_f1[term] = 0.0
        coeffs_f2[term] = 0.0

    # For any terms not in the initial dictionaries, set coefficients to zero
    for name in feature_names:
        coeffs_f1.setdefault(name, 0.0)
        coeffs_f2.setdefault(name, 0.0)

    # Create coefficient lists in order of feature_names
    true_coeffs_f1 = [coeffs_f1[name] for name in feature_names]
    true_coeffs_f2 = [coeffs_f2[name] for name in feature_names]

    return true_coeffs_f1, true_coeffs_f2

def print_parameter_comparison(true_coeffs_f1, true_coeffs_f2, model, labels):
    """
    Compare true coefficients with estimated parameters, including absolute or percentage errors.
    Parameters:
        true_coeffs_f1: True coefficients for dv1/dt.
        true_coeffs_f2: True coefficients for dv2/dt.
        model: Bayesian SINDy model (contains sampled parameters).
        labels: Labels for the parameters.
    """
    print("\n### Parameter Comparison ###\n")

    # Split parameter estimates into dv1/dt and dv2/dt coefficients
    n_coeffs = len(true_coeffs_f1)
    f1_estimates = model.samples[:, :n_coeffs]
    f2_estimates = model.samples[:, n_coeffs:2 * n_coeffs]
    sigma_epsilon_estimates = model.samples[:, -1]

    def format_comparison(true_value, estimate_samples, label):
        median = np.median(estimate_samples)
        std = np.std(estimate_samples)
        if true_value == 0:
            # Use absolute error for parameters with true value of 0
            absolute_error = np.abs(median)
            return f"{label:20}: True = {true_value:.5f}, Estimate = {median:.5f} ± {std:.5f}, Abs. Error = {absolute_error:.5f}"
        else:
            # Use percentage error for nonzero parameters
            percentage_error = 100 * np.abs(median - true_value) / np.abs(true_value)
            return f"{label:20}: True = {true_value:.5f}, Estimate = {median:.5f} ± {std:.5f}, Error = {percentage_error:.2f}%"

    # Print dv1/dt coefficients
    print("### Coefficients for dv1/dt ###")
    for i, name in enumerate(labels[:n_coeffs]):
        print(format_comparison(true_coeffs_f1[i], f1_estimates[:, i], name))

    # Print dv2/dt coefficients
    print("\n### Coefficients for dv2/dt ###")
    for i, name in enumerate(labels[n_coeffs:2 * n_coeffs]):
        print(format_comparison(true_coeffs_f2[i], f2_estimates[:, i], name))

    # Print noise parameter
    print("\n### Noise Parameter ###")
    print(format_comparison(0, sigma_epsilon_estimates, "Sigma_epsilon (Process Noise)"))

def structural_dynamics(state, t, F_t):
    m1 = 1.0  # Mass of first mass
    m2 = 1.0  # Mass of second mass
    k1 = 1.0  # Stiffness of spring connected to ground and mass 1
    c1 = 0.1  # Damping of damper connected to ground and mass 1
    k2 = 1.0  # Stiffness of spring between mass 1 and mass 2
    c2 = 0.1  # Damping of damper between mass 1 and mass 2

    x1, v1, x2, v2 = state

    # Force acting on mass 2
    F = F_t

    # Equations of motion
    a1 = (-k1 * x1 - c1 * v1 + k2 * (x2 - x1) + c2 * (v2 - v1)) / m1
    a2 = (-k2 * (x2 - x1) - c2 * (v2 - v1) + F) / m2

    return [v1, a1, v2, a2]

if __name__ == "__main__":
    # System parameters and data generation
    dt = 0.01
    t = np.arange(0, 5, dt)
    np.random.seed(42)  # For reproducibility

    # Define true model parameters
    m1 = 1.0
    m2 = 1.0
    k1 = 1.0  # Stiffness of spring connected to ground and mass 1
    c1 = 0.1  # Damping of damper connected to ground and mass 1
    k2 = 1.0  # Stiffness of spring between mass 1 and mass 2
    c2 = 0.1  # Damping of damper between mass 1 and mass 2
    mean_force = 1.0    # Constant force acting on mass 2

    # Generate deterministic force acting on mass 2
    F_t = mean_force * np.ones(len(t))

    # Generate training data
    x0 = [0.0, 1.0, 0.5, -0.5]  # [x1_0, v1_0, x2_0, v2_0]
    X = np.zeros((len(t), 4))
    X[0] = x0

    for i in range(len(t) - 1):
        sol = odeint(structural_dynamics, X[i], [t[i], t[i + 1]], args=(F_t[i],))
        X[i + 1] = sol[-1]

    # Calculate derivatives using finite differences
    X_dot = np.gradient(X, dt, axis=0)

    # Fit Bayesian SINDy model (no measurement noise)
    model = BayesianSINDy(n_walkers=256, b=0.01)
    model.fit(X, X_dot, n_steps=5000)

    # After fitting the model
    coefficients = model.samples[:, :-1]  # Exclude sigma_epsilon
    max_coeff = np.max(np.abs(coefficients))
    print("Maximum coefficient magnitude:", max_coeff)

    # Generate predictions with uncertainty for training initial condition
    predictions = model.predict(x0, t)

    # Plot true vs predicted states with uncertainty
    model.plot_true_vs_predicted(t, X, predictions, t_max=2.0)

    # Plot posterior parameter distributions
    model.plot_parameter_distributions()

    # Compute true coefficients
    true_coeffs_f1, true_coeffs_f2 = compute_true_coefficients(
        m1, m2, k1, k2, c1, c2, mean_force, model.feature_names
    )

    # Prepare labels for dv1/dt and dv2/dt coefficients
    labels_f1 = [f'f1_{name}' for name in model.feature_names]
    labels_f2 = [f'f2_{name}' for name in model.feature_names]
    labels = labels_f1 + labels_f2 + ['Sigma_epsilon']

    # Print parameter comparison
    print_parameter_comparison(true_coeffs_f1, true_coeffs_f2, model, labels)

    # Robustness test: Different initial conditions
    print("\n### Robustness Test: Different Initial Conditions ###")
    new_x0 = [1.0, 0.0, -1.0, 0.0]  # New initial conditions [x1, v1, x2, v2]
    X_new = np.zeros((len(t), 4))
    X_new[0] = new_x0

    for i in range(len(t) - 1):
        sol = odeint(structural_dynamics, X_new[i], [t[i], t[i + 1]], args=(F_t[i],))
        X_new[i + 1] = sol[-1]

    # Calculate derivatives for new initial conditions
    X_new_dot = np.gradient(X_new, dt, axis=0)

    # Generate predictions for new initial conditions
    predictions_new = model.predict(new_x0, t)

    # Plot true vs predicted states with uncertainty for new initial conditions
    model.plot_true_vs_predicted(t, X_new, predictions_new, t_max=3.0)

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
