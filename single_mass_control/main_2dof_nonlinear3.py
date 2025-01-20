"""
main_2dof.py

Main script to:
1) Simulate a 2DOF system (the "true" model)
2) Fit a PySINDy model for the deterministic parameters (SINDy)
3) Prune SINDy features and compare them to known true coefficients
4) Plot the discovered model vs. the true system for (a) the same initial condition and
   (b) a new initial condition (robustness check)
5) Optionally, run Bayesian Noise Estimation and MPC
"""

import numpy as np
import matplotlib.pyplot as plt

# Local imports
from simulate_2dof_nonlinear3 import simulate_true, compute_true_coeffs

from sindy_2dof import (
    get_initial_guess_from_pysindy,
    build_expanded_feature_names,
    compare_coeffs,
    prune_sindy_features
)
from bayesian_noise_2dof import BayesianNoiseEstimator
from bayesian_mpc_2dof import run_bayesian_mpc_deterministic_noise
from plot_2dof import plot_mpc_results, plot_true_vs_estimated_model  


###############################################################################
# 2) MAIN SCRIPT
###############################################################################
if __name__ == "__main__":

    # ------------------------------
    # A) System and Simulation Parameters
    # ------------------------------
    m1, m2 = 1.0, 1.0
    k1, k2 = 1.0, 1.0
    c1, c2 = 0.3, 0.3
    a1 , a2 = 0.3, 0.3 # duffy parameters
    theta_0_1, theta_0_2 = 0.5, 0.5
    sigma_epsilon_1, sigma_epsilon_2 = 0.1, 0.1

    dt = 0.005
    t = np.arange(0, 20, dt)
    # np.random.seed(42)  # Uncomment if you want reproducible noise

    # Control input on second mass
    U = 0.5 * np.sin(2 * np.pi * 0.5 * t)

    # Initial conditions for the simulation
    x0 = np.array([1.0, 0.0, 0.5, -0.2])

    # Noise arrays
    noise_array_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
    noise_array_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))

    # ------------------------------
    # B) Simulate the "True" 2DOF System
    # ------------------------------
    X_true, X_dot_true = simulate_true(
        m1, m2, c1, c2, k1, k2, 
        a1, a2,
        theta_0_1, theta_0_2, 
        x0, t, U, 
        noise_array_1, noise_array_2
    )

    # ------------------------------
    # C) Fit PySINDy (Deterministic Part)
    # ------------------------------
    initial_guess, feat_names, fitted_sindy_model = get_initial_guess_from_pysindy(
        X_true, X_dot_true, U, t,
        rows_for_coeffs=(1, 3),
        poly_degree=3,
        include_bias=True,
        include_interactions=False
    )

    # Check shape
    det_params_matrix = fitted_sindy_model.coefficients()
    det_feature_names = fitted_sindy_model.get_feature_names()
    print("Deterministic parameters matrix shape:\n", det_params_matrix.shape)

    # ------------------------------
    # D) Prune SINDy Features + Compare
    # ------------------------------
    pruned_coeff_matrix, pruned_feature_names, active_idx = prune_sindy_features(
        fitted_sindy_model,
        rows_for_coeffs=(1,3),
        tol=1e-6
    )
    print("Pruned coeff matrix shape:", pruned_coeff_matrix.shape)

    # Build expanded names to compare with "true" coeffs
    expanded_names = build_expanded_feature_names(feat_names)
    
    true_coeffs = compute_true_coeffs(
        m1, m2, c1, c2, k1, k2, 
        theta_0_1, theta_0_2, 
        sigma_epsilon_1, sigma_epsilon_2,
        feat_names,
        a1, a2,
    )
    # Show pruned comparison table
    df_pruned = compare_coeffs(true_coeffs, initial_guess, expanded_names,
                               active_feature_indices=active_idx)
    print(df_pruned)

    # ------------------------------
    # E) Plot: True vs. Discovered Model (Same IC)
    # ------------------------------
    X_true_no_noise, X_dot_true_no_noise = simulate_true(
        m1, m2, c1, c2, k1, k2, 
        a1, a2,
        theta_0_1, theta_0_2 ,
        x0, t, U, 
        np.zeros(len(t)), np.zeros(len(t))
    )
    
    plot_true_vs_estimated_model(
        t,
        X_true_no_noise,  # the "true" system states
        x0,
        U,
        fitted_sindy_model,
        title_prefix="In-Sample"
    )

    # ------------------------------
    # F) Robustness Check (New IC)
    # ------------------------------
    x0_robust = np.array([-0.5, 0.3, 1.2, 0.0])  # Some new initial condition
    # Re-simulate "true" system from this new x0_robust

    X_true_robust_no_noise, X_dot_true_robust_no_noise = simulate_true(
        m1, m2, c1, c2, k1, k2,
        a1, a2,
        theta_0_1, theta_0_2,
        x0_robust, t, U,
        np.zeros(len(t)), np.zeros(len(t))
    )

    # Plot: True vs discovered model from that new IC
    plot_true_vs_estimated_model(
        t,
        X_true_robust_no_noise,
        x0_robust,
        U,
        fitted_sindy_model,
        title_prefix="Robustness Check (New IC)"
    )

    # ------------------------------
    # G) Bayesian Noise Estimation (Optional)
    # ------------------------------
    bayes_noise_model = BayesianNoiseEstimator(
        det_sindy_model=fitted_sindy_model,
        rows_for_coeffs=(1, 3),
        n_walkers=20,
    )
    bayes_noise_model.fit(X_true, X_dot_true, U=U, t=t, n_steps=200, initial_sigma=(0.0, 0.0))
    sigma_samples = bayes_noise_model.get_sigma_samples()
    print("Posterior samples for [sigma1, sigma2]:", sigma_samples.shape)

    # Plot the noise posterior
    plt.figure()
    plt.hist(sigma_samples[:,0], bins=30, alpha=0.5, label='sigma_epsilon_1')
    plt.hist(sigma_samples[:,1], bins=30, alpha=0.5, label='sigma_epsilon_2')
    plt.legend()
    plt.title("Posterior Distributions for Noise Parameters")
    plt.show()

    # ------------------------------
    # H) Prepare & Run MPC (Optional)
    # ------------------------------
    # Uncontrolled system for comparison
    noise_array_unc_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
    noise_array_unc_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))
    U_no_control = np.zeros(len(t))

    X_uncontrolled, X_dot_uncontrolled = simulate_true(
        m1, m2, c1, c2, k1, k2, 
        a1, a2,
        theta_0_1, theta_0_2, 
        x0, t, U_no_control, 
        noise_array_unc_1, 
        noise_array_unc_2
    )

    # MPC parameters
    N = 15
    Q = np.diag([100, 1, 100, 1])
    R = 0.1
    u_max, u_min = 1.0, -1.0
    x_ref = np.array([0.0, 0.0, 0.0, 0.0])

    # Sample from posterior
    n_mpc_samples = 1
    idxs = np.random.choice(len(sigma_samples), size=n_mpc_samples, replace=False)
    sigma_draws = sigma_samples[idxs]

    # Run Bayesian MPC
    X_all, U_all, X_mean, X_std, U_mean, U_std = run_bayesian_mpc_deterministic_noise(
        pruned_feature_names,
        pruned_coeff_matrix,
        fitted_sindy_model, 
        sigma_draws, 
        x0, t, Q, R, N, u_min, u_max, x_ref
    )

    # Plot MPC vs. Uncontrolled
    plot_mpc_results(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
