"""
main_2dof.py

Main script to:
1) Simulate a 2DOF system (the "true" model) with two distinct inputs U1(t), U2(t)
2) Fit a PySINDy model for the deterministic parameters (SINDy)
3) Prune SINDy features and compare them to known true coefficients
4) Plot the discovered model vs. the true system for (a) the same initial condition and
   (b) a new initial condition (robustness check)
5) Optionally, run Bayesian Noise Estimation and MPC
"""

import numpy as np
import matplotlib.pyplot as plt

# Local imports
from simulate_2dof_linear_double_control import simulate_true, compute_true_coeffs

from sindy_2dof import (
    get_initial_guess_from_pysindy,
    build_expanded_feature_names,
    compare_coeffs,
    prune_sindy_features
)
from bayesian_noise_2dof import BayesianNoiseEstimator
from bayesian_mpc_2dof_double_control import run_bayesian_mpc_deterministic_noise
from plot_2dof_double_control import (
    plot_mpc_results, plot_true_vs_estimated_model, plot_mpc_results_phase
    )

###############################################################################
# MAIN SCRIPT
###############################################################################
if __name__ == "__main__":

    # ------------------------------
    # A) System and Simulation Parameters
    # ------------------------------
    m1, m2 = 1.0, 1.0
    k1, k2 = 1.0, 1.0
    c1, c2 = 0.3, 0.3
    theta_0_1, theta_0_2 = 0.5, 0.5
    sigma_epsilon_1, sigma_epsilon_2 = 0.05, 0.05

    dt = 0.005
    t = np.arange(0, 15, dt)

    # ------------------------------
    # B) Define Two Independent Controls U1(t), U2(t)
    # ------------------------------
    # Example: two different sinusoidal inputs
    U1 = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    U2 = 0.3 * np.cos(2 * np.pi * 0.7 * t)
    # Combine into a single 2D array U of shape (len(t), 2)
    U = np.column_stack([U1, U2])

    # Initial conditions for the simulation
    x0 = np.array([1.0, 0.0, 0.5, -0.2])

    # Noise arrays
    noise_array_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
    noise_array_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))

    # ------------------------------
    # C) Simulate the "True" 2DOF System
    # ------------------------------
    X_true, X_dot_true = simulate_true(
        m1, m2, c1, c2, k1, k2, 
        theta_0_1, theta_0_2, 
        x0, t, U,  # <-- now passing the 2D control
        noise_array_1, noise_array_2
    )

    # ------------------------------
    # D) Fit PySINDy (Deterministic Part)
    # ------------------------------
    # IMPORTANT: pass the 2D control array as 'u=U'
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
    # E) Prune SINDy Features + Compare
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
        feat_names
    )
    # Show pruned comparison table
    df_pruned = compare_coeffs(true_coeffs, initial_guess, expanded_names,
                                active_feature_indices=None)
    print(df_pruned)

    # ------------------------------
    # F) Plot: True vs. Discovered Model (Same IC)
    # ------------------------------
    X_true_no_noise, X_dot_true_no_noise = simulate_true(
        m1, m2, c1, c2, k1, k2, 
        theta_0_1, theta_0_2,
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
    # G) Robustness Check (New IC)
    # ------------------------------
    x0_robust = np.array([-0.5, 0.3, 1.2, 0.0])  # Some new initial condition

    X_true_robust_no_noise, X_dot_true_robust_no_noise = simulate_true(
        m1, m2, c1, c2, k1, k2,
        theta_0_1, theta_0_2,
        x0_robust, t, U,
        np.zeros(len(t)), np.zeros(len(t))
    )

    plot_true_vs_estimated_model(
        t,
        X_true_robust_no_noise,
        x0_robust,
        U,
        fitted_sindy_model,
        title_prefix="Robustness Check (New IC)"
    )

    # ------------------------------
    # H) Bayesian Noise Estimation (Optional)
    # ------------------------------
    bayes_noise_model = BayesianNoiseEstimator(
        det_sindy_model=fitted_sindy_model,
        rows_for_coeffs=(1, 3),
        n_walkers=16,
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
    # I) Prepare & Run MPC (Optional)
    # ------------------------------
    # For an "uncontrolled" reference run, just pass zeros in U
    U_no_control = np.zeros((len(t), 2))
    noise_array_unc_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
    noise_array_unc_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))

    X_uncontrolled, X_dot_uncontrolled = simulate_true(
        m1, m2, c1, c2, k1, k2, 
        theta_0_1, theta_0_2, 
        x0_robust, t, U_no_control, 
        noise_array_unc_1, 
        noise_array_unc_2
    )

    # MPC parameters
    N = 10
    Q = np.diag([100, 1, 100, 1])
    R = 0.01  # could also use a matrix, e.g. np.diag([0.1, 0.1]) for u1,u2
    u_max, u_min = 3.0, -3.0
    x_ref = [0,0,0,0]

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
    plot_mpc_results_phase(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
