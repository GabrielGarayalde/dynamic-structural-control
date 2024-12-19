# Bayesian SINDy with MPC - 2DOF System
# Now with Bayesian MPC, Comparison between Uncontrolled and MPC Controlled Trajectories,
# and Robustness Test with Multiple Initial Conditions

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import emcee
import corner
from tqdm import tqdm
from scipy.optimize import minimize

# ------------------------------
# 1. Feature Library and True Coefficients
# ------------------------------

def build_library(X, U):
    """
    Build a library of candidate features from the state variables and control input u.
    X: (n_samples, 4) with columns [x1, v1, x2, v2]
    U: (n_samples,) control input applied to the second mass.
    """
    x1 = X[:, 0]
    v1 = X[:, 1]
    x2 = X[:, 2]
    v2 = X[:, 3]
    u = U

    # Uncomment the following lines to include non-linear terms in the feature library
    # features = [x1, v1, x2, v2, x1**2, x2**2, u]
    # feature_names = ['x1', 'v1', 'x2', 'v2', 'x1^2', 'x2^2', 'u']
    
    # Current feature library without non-linear terms
    features = [x1, v1, x2, v2]
    features.extend([x1**2, x2**2, 
                     v1**2, v2**2,
                      # x1*v1, x1*v2,
                     # x2*v1, x2*v2,
                     u
                     ])

    feature_names = ['x1', 'v1', 'x2', 'v2']
    feature_names.extend(['x1^2', 'x2^2',
                          'v1^2', 'v2^2',
                           # 'x1v1', 'x1v2',
                          # 'x2v1', 'x2v2',
                          'u'])
    Theta = np.column_stack(features)
    return Theta, feature_names

def compute_true_coeffs(m1, m2, k1, k2, c1, c2, theta_0, sigma_epsilon, feature_names):
    """
    Compute the "true" coefficients including the control input 'u' for reference.
    (These are known values used for parameter comparison, not for plotting the true trajectory.)
    """
    # Coefficients for v1_dot
    c_v1_x1 = -k1/m1
    c_v1_v1 = -c1/m1
    c_v1_x2 = k1/m1
    c_v1_v2 = c1/m1
    c_v1_x1_sq = 0
    c_v1_x2_sq = 0
    c_v1_v1_sq = 0
    c_v1_v2_sq = 0
    # c_v1_x1v1 = 0
    # c_v1_x1v2 = 0
    # c_v1_x2v1 = 0
    # c_v1_x2v2 = 0
    c_v1_u = 0

    # Coefficients for v2_dot
    c_v2_x1 = k1/m2
    c_v2_v1 = c1/m2
    c_v2_x2 = -(k1 + k2)/m2
    c_v2_v2 = -(c1 + c2)/m2
    c_v2_x1_sq = 0
    c_v2_x2_sq = 0
    c_v2_v1_sq = 0
    c_v2_v2_sq = 0
    # c_v2_x1v1 = 0
    # c_v2_x1v2 = 0
    # c_v2_x2v1 = 0
    # c_v2_x2v2 = 0
    c_v2_u = 1.0/m2

    true_coeffs = [
        theta_0,
        c_v1_x1, c_v1_v1, c_v1_x2, c_v1_v2, 
        c_v1_x1_sq, c_v1_x2_sq,  
        c_v1_v1_sq, c_v1_v2_sq,
        # c_v1_x1v1, c_v1_x1v2,
        # c_v1_x2v1, c_v1_x2v2,
        c_v1_u,   
        c_v2_x1, c_v2_v1, c_v2_x2, c_v2_v2, 
        c_v2_x1_sq, c_v2_x2_sq,
        c_v2_v1_sq, c_v2_v2_sq,
        # c_v2_x1v1, c_v2_x1v2,
        # c_v2_x2v1, c_v2_x2v2,
        c_v2_u,  
        sigma_epsilon
    ]
    return true_coeffs

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

    def log_likelihood(self, theta, X, X_dot, U):
        theta_0 = theta[0]
        sigma_epsilon = theta[-1]

        Theta_matrix, _ = build_library(X, U)
        M = Theta_matrix.shape[1]
        coeffs_v1 = theta[1:1+M]
        coeffs_v2 = theta[1+M:1+2*M]

        # Predicted derivatives
        v1_dot_pred = Theta_matrix @ coeffs_v1
        v2_dot_pred = theta_0 + (Theta_matrix @ coeffs_v2)

        # Known relations
        x1_dot_pred = X[:,1]
        x2_dot_pred = X[:,3]

        # Uncertainties
        sigma_x = 1e-8
        sigma_v1 = 1e-8
        sigma_v2 = sigma_epsilon

        def normal_loglike(y, y_pred, sigma):
            return -0.5*np.sum(((y - y_pred)/sigma)**2 + np.log(2*np.pi*sigma**2))

        log_like_x1 = normal_loglike(X_dot[:,0], x1_dot_pred, sigma_x)
        log_like_v1 = normal_loglike(X_dot[:,1], v1_dot_pred, sigma_v1)
        log_like_x2 = normal_loglike(X_dot[:,2], x2_dot_pred, sigma_x)
        log_like_v2 = normal_loglike(X_dot[:,3], v2_dot_pred, sigma_v2)

        return log_like_x1 + log_like_v1 + log_like_x2 + log_like_v2

    def log_probability(self, theta, X, X_dot, U):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta, X, X_dot, U)
        return lp + ll

    def fit(self, X, X_dot, U, n_steps=2000):
        Theta_matrix, feature_names = build_library(X, U)
        self.feature_names = feature_names
        M = Theta_matrix.shape[1]

        # Total params = 1(theta_0) + 2*M(coeffs) + 1(sigma_epsilon)
        n_params = 2*M + 2
        self.n_params = n_params

        initial_guess = np.zeros(n_params)
        initial_guess[-1] = 0.1  # sigma_epsilon > 0

        pos = initial_guess + 0.01*np.random.randn(self.n_walkers, n_params)
        pos[:, -1] = np.abs(pos[:, -1]) + 1e-3

        sampler = emcee.EnsembleSampler(
            self.n_walkers, n_params, self.log_probability,
            args=(X, X_dot, U)
        )

        print("Running MCMC (burn-in)...")
        state = sampler.run_mcmc(pos, 500, progress=True)
        sampler.reset()
        print("Running MCMC (production)...")
        sampler.run_mcmc(state, n_steps, progress=True)

        self.samples = sampler.get_chain(discard=500, thin=10, flat=True)
        return self

    
    def predict(self, x0, t, U, n_samples=100):
        if self.samples is None:
            raise ValueError("Model not fitted yet.")
    
        idx = np.random.randint(len(self.samples), size=n_samples)
        predictions = []
    
        def system(state, t_val, theta_params, u_val, noise):
            x1, v1, x2, v2 = state
            M = (len(theta_params)-2)//2
            theta_0 = theta_params[0]
            
            coeffs_v1 = theta_params[1:1+M]
            coeffs_v2 = theta_params[1+M:1+2*M]
    
            # Efficient feature computation without loops
            Theta_point = np.array([x1, v1, x2, v2, 
                                    x1**2, x2**2, 
                                    v1**2, v2**2, 
                                    # x1*v1, x1*v2,
                                    # x2*v1, x2*v2,  # Uncomment if needed
                                    u_val])  # [x1, v1, x2, v2, x1^2, x2^2, v1^2, v2^2, x1*v1, x1*v2, u]
    
            v1_dot = Theta_point @ coeffs_v1
            v2_dot = theta_0 + (Theta_point @ coeffs_v2) + noise  # Add noise to v2_dot
    
            x1_dot = v1
            x2_dot = v2
            return [x1_dot, v1_dot, x2_dot, v2_dot]
    
        print("Generating predictions:")
        for theta in tqdm(self.samples[idx], desc="Generating predictions"):
            theta_0 = theta[0]
            sigma_epsilon = theta[-1]
    
            # Generate noise with mean 0 and std sigma_epsilon
            noise_force = np.random.normal(0, sigma_epsilon, size=len(t))
    
            X_pred = np.zeros((len(t), 4))
            X_pred[0] = x0
            for i in range(len(t)-1):
                sol = odeint(system, X_pred[i], [t[i], t[i+1]], args=(theta, U[i], noise_force[i]))
                X_pred[i+1] = sol[-1]
            predictions.append(X_pred)
    
        return np.array(predictions)


# ------------------------------
# 3. Parameter Diagnostics and Comparison
# ------------------------------

def plot_true_vs_estimated_uncontrolled_for_ic(t, X_true_uncontrolled, X_estimated_uncontrolled_simulations, initial_condition_label="IC"):
    """
    Plot the system behavior under u=0 using True Parameters vs Estimated Parameters with Uncertainty.
    """
    pred_mean = np.mean(X_estimated_uncontrolled_simulations, axis=0)
    pred_std = np.std(X_estimated_uncontrolled_simulations, axis=0)
    states = ['x1', 'v1', 'x2', 'v2']
    fig, axs = plt.subplots(4,1,figsize=(10,16))
    for i in range(4):
        axs[i].plot(t, X_true_uncontrolled[:,i], 'b-', label='True Params (u=0)')
        axs[i].plot(t, pred_mean[:,i], 'r--', label='Estimated Params Mean (u=0)')
        axs[i].fill_between(t, pred_mean[:,i]-2*pred_std[:,i], pred_mean[:,i]+2*pred_std[:,i],
                            color='r', alpha=0.2, label='95% CI')
        axs[i].set_xlabel('Time [s]')
        axs[i].set_ylabel(states[i])
        axs[i].legend()
        axs[i].grid(True)
        axs[i].set_title(f'{initial_condition_label}: {states[i]} under u=0: True vs Estimated')
    plt.tight_layout()
    plt.show()

def plot_parameter_distributions(model):
    """
    Plot the posterior distributions of the identified parameters using corner plots.
    """
    corner.corner(model.samples, quantiles=[0.16,0.5,0.84], show_titles=True, title_kwargs={"fontsize":12})
    plt.show()

def print_parameter_comparison(true_values, model, feature_names):
    """
    Compare true coefficients with estimated parameters and report errors and uncertainties.
    """
    print("\n### Parameter Comparison ###\n")
    M = len(feature_names)
    derivatives = ['v1_dot', 'v2_dot']
    param_labels = ['theta_0']
    for deriv in derivatives:
        for feat in feature_names:
            param_labels.append(f'c_{deriv}_{feat}')
    param_labels.append('sigma_epsilon')

    def format_comparison(tv, est, lbl):
        median = np.median(est)
        std = np.std(est)
        if lbl == 'sigma_epsilon':
            abs_err = np.abs(median - tv)
            return f"{lbl:25}: True={tv:.5f}, Est={median:.5f}±{std:.5f}, AbsErr={abs_err:.5f}"
        if tv == 0:
            err = np.abs(median)
            unc = std
            return f"{lbl:25}: True={tv:.5f}, Est={median:.5f}±{std:.5f}, AbsErr={err:.5f}, Unc={unc:.5f}"
        else:
            perr = 100 * np.abs(median - tv) / np.abs(tv)
            unc = 100 * std / np.abs(tv)
            return f"{lbl:25}: True={tv:.5f}, Est={median:.5f}±{std:.5f}, Err={perr:.2f}%, Unc={unc:.2f}%"

    if len(true_values) != len(param_labels):
        print("Warning: Number of true values does not match number of parameters.")
        print(f"True values length: {len(true_values)}, Parameter labels length: {len(param_labels)}")
        min_len = min(len(true_values), len(param_labels))
        true_values = true_values[:min_len]
        param_labels = param_labels[:min_len]

    for i, lbl in enumerate(param_labels):
        tv = true_values[i]
        est = model.samples[:, i]
        print(format_comparison(tv, est, lbl))

# ------------------------------
# 4. Identified Model Dynamics and MPC
# ------------------------------

def identified_model_step(x, u_val, dt, theta_0, coeffs_v1, coeffs_v2):
    """
    Perform one step of state propagation using the identified model parameters for the 2DOF system.
    """
    x1, v1, x2, v2 = x
    
    Theta_point = np.column_stack([x1, v1, x2, v2, 
                                   x1**2, x2**2, 
                                   v1**2, v2**2, 
                                    # x1*v1, x1*v2,
                                   # x2*v1, x2*v2,
                                   u_val])  # [x1, v1, x2, v2, u]

    v1_dot = (Theta_point @ coeffs_v1)[0]
    v2_dot = theta_0 + (Theta_point @ coeffs_v2)[0]

    x1_next = x1 + dt*v1
    v1_next = v1 + dt*v1_dot
    x2_next = x2 + dt*v2
    v2_next = v2 + dt*v2_dot
    return np.array([x1_next, v1_next, x2_next, v2_next])

def run_bayesian_mpc(model, x0, t, U, N=20, Q=np.diag([100, 1, 100, 1]), R=0.1, 
                     u_max=10.0, u_min=-10.0, x_ref=np.array([0.0,0.0,0.0,0.0]),
                     n_mpc_samples=100):
    """
    Run MPC simulations incorporating model uncertainty from Bayesian SINDy.
    Compare Uncontrolled (u=0) vs MPC Mean trajectory.

    Returns:
    X_simulations, U_simulations, X_mean, X_std, U_mean, U_std
    """
    dt = t[1] - t[0]

    sampled_params = model.samples[np.random.choice(len(model.samples), size=n_mpc_samples, replace=False)]

    X_simulations = np.zeros((n_mpc_samples, len(t), 4))
    U_simulations = np.zeros((n_mpc_samples, len(t)-1))

    
    print("Running Bayesian MPC simulations...")
    for s in tqdm(range(n_mpc_samples), desc="MPC Sim"):
        params = sampled_params[s]
        theta_0_id = params[0]
        # M=5 features per equation: [x1, v1, x2, v2, u]
        M=9
        coeffs_v1_id = params[1:1+M]
        coeffs_v2_id = params[1+M:1+2*M]

        x_current = x0.copy()
        X_sim = [x_current]
        U_seq = []

        for idx in range(len(t)-1):
            # Define MPC optimization problem
            def mpc_cost(u_sequence):
                cost = 0.0
                x_pred = x_current.copy()
                for u_k in u_sequence[:N]:
                    error = x_pred - x_ref
                    cost += error.T @ Q @ error
                    cost += R*(u_k**2)
                    x_pred = identified_model_step(x_pred, u_k, dt, theta_0_id, coeffs_v1_id, coeffs_v2_id)
                return cost

            # Initial guess for control sequence
            u_init = np.zeros(N)
            # Bounds for control inputs
            bounds = [(u_min, u_max)] * N

            # Optimize control sequence
            res = minimize(
                mpc_cost,
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

            U_seq.append(u_opt)
            x_next = identified_model_step(x_current, u_opt, dt, theta_0_id, coeffs_v1_id, coeffs_v2_id)
            X_sim.append(x_next)
            x_current = x_next

        X_sim = np.array(X_sim)
        U_sim = np.array(U_seq)

        X_simulations[s, :len(X_sim), :] = X_sim
        U_simulations[s, :len(U_sim)] = U_sim

    # Compute statistics
    X_mean = np.mean(X_simulations, axis=0)
    X_std = np.std(X_simulations, axis=0)
    U_mean = np.mean(U_simulations, axis=0)
    U_std = np.std(U_simulations, axis=0)

    return X_simulations, U_simulations, X_mean, X_std, U_mean, U_std

# ------------------------------
# 5. Plotting Uncontrolled vs. MPC Results
# ------------------------------

def plot_uncontrolled_vs_mpc(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot the Uncontrolled Trajectory vs MPC Mean Trajectory with uncertainty.
    Also plot the MPC control input vs. uncontrolled (u=0).
    """
    states = ['x1', 'v1', 'x2', 'v2']
    fig, axs = plt.subplots(5, 1, figsize=(14, 24))

    # Plot states: Compare X_uncontrolled (u=0) with MPC Mean ± uncertainty
    for i in range(4):
        axs[i].plot(t, X_uncontrolled[:, i], 'g-', label='Uncontrolled (u=0)')
        axs[i].plot(t, X_mean[:, i], 'r-', label='MPC Mean')
        axs[i].fill_between(t, X_mean[:, i]-2*X_std[:, i], X_mean[:, i]+2*X_std[:, i],
                            color='r', alpha=0.2, label='95% CI')
        axs[i].set_xlabel('Time [s]')
        axs[i].set_ylabel(states[i])
        axs[i].legend()
        axs[i].grid(True)
        axs[i].set_title(f'Comparison of {states[i]}: Uncontrolled vs MPC')

    # Plot control input
    axs[4].plot(t[:-1], U_mean, 'b-', label='MPC Mean Control Input')
    axs[4].fill_between(t[:-1], U_mean - 2*U_std, U_mean + 2*U_std,
                       color='b', alpha=0.2, label='95% CI')
    axs[4].axhline(0, color='k', linestyle='--', label='Uncontrolled u = 0')
    axs[4].set_xlabel('Time [s]')
    axs[4].set_ylabel('Control Input u')
    axs[4].legend()
    axs[4].grid(True)
    axs[4].set_title('MPC Control Input with Uncertainty and Uncontrolled Case')

    plt.tight_layout()
    plt.show()

# ------------------------------
# 6. Main Execution Block
# ------------------------------

if __name__ == "__main__":
    # ------------------------------
    # 6.1. System Parameters
    # ------------------------------
    m1 = 1.0
    m2 = 1.0
    k1 = 1.0
    k2 = 1.0
    c1 = 0.4
    c2 = 0.2
    theta_0 = 0.5
    sigma_epsilon = 0.05

    dt = 0.005
    t = np.arange(0, 20, dt)  # Extended to 60 seconds for better visualization
    np.random.seed(42)

    # ------------------------------
    # 6.2. Control Input Definition
    # ------------------------------
    U = 0.5 * np.sin(2 * np.pi * 0.5 * t)  # Example control input applied to the second mass

    # ------------------------------
    # 6.3. System Dynamics Simulation with Control
    # ------------------------------
    def structural_dynamics(state, t_val, u_val):
        """
        Define the system's structural dynamics.
        """
        x1, v1, x2, v2 = state
        dx1 = v1
        dv1 = (-c1*(v1 - v2) - k1*(x1 - x2))/m1
        dx2 = v2
        dv2 = (theta_0 + u_val - c1*(v2 - v1) - k1*(x2 - x1) - c2*v2 - k2*x2)/m2
        return [dx1, dv1, dx2, dv2]

    # Define multiple initial conditions for robustness testing
    initial_conditions = [
        [1.0, 0.0, 0.5, 0.0],   # Original initial condition
        [0.8, -0.2, 0.6, 0.1],
        # [1.2, 0.3, 0.4, -0.1],
        # [0.5, -0.5, 0.7, 0.2],
        # [1.5, 0.4, 0.3, -0.2]
    ]

    # Select the first initial condition for system simulation with control
    x0 = initial_conditions[0]  # Initial state: [x1, v1, x2, v2]

    # Simulate system dynamics with control input U
    X = np.zeros((len(t),4))
    X[0] = x0
    X_dot = np.zeros((len(t),4))
    for i in range(len(t)-1):
        sol = odeint(structural_dynamics, X[i], [t[i], t[i+1]], args=(U[i],))
        X[i+1] = sol[-1]
        X_dot[i] = structural_dynamics(X[i], t[i], U[i])
    X_dot[-1] = structural_dynamics(X[-1], t[-1], U[-1])

    # ------------------------------
    # 6.4. Compute True Coefficients and Fit Bayesian SINDy
    # ------------------------------
    Theta_example, feature_names_example = build_library(X, U)
    true_coeffs = compute_true_coeffs(m1, m2, k1, k2, c1, c2, theta_0, sigma_epsilon, feature_names_example)

    model = BayesianSINDy(n_walkers=1600, b=0.1)
    model.fit(X, X_dot, U, n_steps=1000)

    # (Optional) Parameter Diagnostics
    plot_parameter_distributions(model)
    print_parameter_comparison(true_coeffs, model, model.feature_names)

    # ------------------------------
    # 6.5. Simulate the True System (with and without control)
    # ------------------------------
    def simulate_true(model, x0, t, U):
        """
        Simulate the true system dynamics.
        """
        X_true = np.zeros((len(t), 4))
        X_true[0] = x0
        for i in range(len(t)-1):
            sol = odeint(structural_dynamics, X_true[i], [t[i], t[i+1]], args=(U[i],))
            X_true[i+1] = sol[-1]
        return X_true

    # ------------------------------
    # 6.6. Simulate and Plot True vs Estimated Parameters under u=0
    # ------------------------------
    def simulate_estimated_parameters_uncontrolled(model, t, x0, n_samples=50):
        """
        Simulate the system with u=0 using estimated parameters sampled from the posterior.
        """
        dt = t[1] - t[0]
        sampled_params = model.samples[np.random.choice(len(model.samples), size=n_samples, replace=False)]
        simulations = []

        Theta_matrix, _ = build_library(X, U)
        M = Theta_matrix.shape[1]
        
        print("Simulating system under estimated parameters (u=0)...")
        for s in tqdm(range(n_samples), desc="Estimated Params Sim"):
            params = sampled_params[s]
            theta_0_id = params[0]
            coeffs_v1_id = params[1:1+M]
            coeffs_v2_id = params[1+M:1+2*M]
            
            x_current = x0.copy()
            X_sim = [x_current]

            for i in range(len(t)-1):
                u_val = 0.0  # u=0 for uncontrolled
                x_next = identified_model_step(x_current, u_val, dt, theta_0_id, coeffs_v1_id, coeffs_v2_id)
                X_sim.append(x_next)
                x_current = x_next

            simulations.append(np.array(X_sim))

        simulations = np.array(simulations)
        return simulations

    def simulate_true_uncontrolled(model, x0, t):
        """
        Simulate the true system dynamics under u=0.
        """
        U_no_control = np.zeros(len(t))
        X_true_uncontrolled = simulate_true(model, x0, t, U_no_control)
        return X_true_uncontrolled

    def plot_true_vs_estimated_uncontrolled_for_ic(t, X_true_uncontrolled, X_estimated_uncontrolled_simulations, initial_condition_label="IC"):
        """
        Plot the system behavior under u=0 using True Parameters vs Estimated Parameters with Uncertainty.
        """
        pred_mean = np.mean(X_estimated_uncontrolled_simulations, axis=0)
        pred_std = np.std(X_estimated_uncontrolled_simulations, axis=0)
        states = ['x1', 'v1', 'x2', 'v2']
        fig, axs = plt.subplots(4,1,figsize=(10,16))
        for i in range(4):
            axs[i].plot(t, X_true_uncontrolled[:,i], 'b-', label='True Params (u=0)')
            axs[i].plot(t, pred_mean[:,i], 'r--', label='Estimated Params Mean (u=0)')
            axs[i].fill_between(t, pred_mean[:,i]-2*pred_std[:,i], pred_mean[:,i]+2*pred_std[:,i],
                                color='r', alpha=0.2, label='95% CI')
            axs[i].set_xlabel('Time [s]')
            axs[i].set_ylabel(states[i])
            axs[i].legend()
            axs[i].grid(True)
            axs[i].set_title(f'{initial_condition_label}: {states[i]} under u=0: True vs Estimated')
        plt.tight_layout()
        plt.show()

    # ------------------------------
    # 6.7. Plot True vs Estimated Parameters under u=0 for Each Initial Condition
    # ------------------------------
    for idx, ic in enumerate(initial_conditions):
        # True system simulation under u=0
        X_true_ic = simulate_true_uncontrolled(model, ic, t)

        # Estimated system simulation under u=0
        X_estimated_ic = simulate_estimated_parameters_uncontrolled(model, t, ic, n_samples=25)

        # Plot comparison
        plot_true_vs_estimated_uncontrolled_for_ic(t, X_true_ic, X_estimated_ic, initial_condition_label=f"IC{idx+1}")

    # ------------------------------
    # 6.8. Run Bayesian MPC
    # ------------------------------
    N = 20                      # Prediction horizon
    Q = np.diag([100, 1, 100, 1])  # State weights: penalize displacements and velocities
    R = 0.1                     # Control weight: penalize large control inputs
    u_max = 1.0                 # Maximum control input
    u_min = -1.0                # Minimum control input
    x_ref = np.array([0.0, 0.0, 0.0, 0.0])  # Reference state: rest position and zero velocity
    x0_mpc = np.array(initial_conditions[1])      # Initial state for MPC
    n_mpc_samples = 3          # Number of posterior samples to simulate

    X_simulations, U_simulations, X_mean, X_std, U_mean, U_std = run_bayesian_mpc(
        model, x0_mpc, t, U, N=N, Q=Q, R=R, 
        u_max=u_max, u_min=u_min, 
        x_ref=x_ref, n_mpc_samples=n_mpc_samples
    )

    # ------------------------------
    # 6.9. Plot Uncontrolled vs MPC Results
    # ------------------------------
    X_uncontrolled = simulate_true_uncontrolled(model, x0, t)
    plot_uncontrolled_vs_mpc(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
