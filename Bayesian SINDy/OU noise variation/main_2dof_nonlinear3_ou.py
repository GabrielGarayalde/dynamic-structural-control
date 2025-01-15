# main_2dof_nonlinear_ou.py

import numpy as np
import matplotlib.pyplot as plt

# Local imports (the new file you just created!)
from simulate_2dof_nonlinear3_ou import simulate_true, compute_true_coeffs

# Same SINDy and plotting utilities as before
from sindy_2dof import (
    get_initial_guess_from_pysindy,
    build_expanded_feature_names,
    compare_coeffs,
    prune_sindy_features
)
from bayesian_noise_2dof import BayesianNoiseEstimator
from bayesian_mpc_2dof import run_bayesian_mpc_deterministic_noise
from plot_2dof import plot_mpc_results, plot_true_vs_estimated_model

if __name__ == "__main__":

    # A) System + Simulation Parameters
    m1, m2 = 1.0, 1.0
    k1, k2 = 1.0, 1.0
    c1, c2 = 0.3, 0.3
    alpha1, alpha2 = 0.2, 0.0  # Duffing parameters
    theta_0_1, theta_0_2 = 0.5, 0.5

    # Noise parameters
    sigma_epsilon_1, sigma_epsilon_2 = 0.1, 0.1
    dt = 0.005
    t = np.arange(0, 20, dt)
    # np.random.seed(42)  # Uncomment if you want reproducible noise

    # Control input
    U = 0.5 * np.sin(2*np.pi*0.5*t)

    # Initial conditions
    x0 = np.array([1.0, 0.0, 0.5, -0.2])

    # We want "ou" noise:
    noise_type = 'ou'
    # or noise_type = None if you want to test zero noise, etc.

    # B) Simulate the "True" 2DOF Nonlinear System with OU noise
    X_true, X_dot_true = simulate_true(
        m1, m2, c1, c2, k1, k2,
        alpha1, alpha2,
        theta_0_1, theta_0_2,
        x0, t, U,
        noise_type=noise_type,
        sigma_epsilon_1=sigma_epsilon_1,
        sigma_epsilon_2=sigma_epsilon_2,
        theta_ou=1.0,    # OU mean-reversion parameter
        mu_ou=0.0,       
        x0_ou1=0.0,
        x0_ou2=0.0,
        seed=None
    )

    # C) Fit SINDy (Deterministic Part)
    initial_guess, feat_names, fitted_sindy_model = get_initial_guess_from_pysindy(
        X_true, X_dot_true, U, t,
        rows_for_coeffs=(1, 3),
        poly_degree=3,
        include_bias=True,
        include_interactions=True
    )

    # e.g. if your true system is x^3 duffing, you might want poly_degree=3
    det_params_matrix = fitted_sindy_model.coefficients()
    det_feature_names = fitted_sindy_model.get_feature_names()
    print("Deterministic parameters matrix shape:", det_params_matrix.shape)

    # D) Prune & Compare
    pruned_coeff_matrix, pruned_feature_names, active_idx = prune_sindy_features(
        fitted_sindy_model,
        rows_for_coeffs=(1,3),
        tol=1e-6
    )
    print("Pruned coeff matrix shape:", pruned_coeff_matrix.shape)

    # Build expanded names to compare with "true" coeffs
    expanded_names = build_expanded_feature_names(feat_names)

    # (Optional) compute the "true" coefficients, if your code needs them:
    true_coeffs = compute_true_coeffs(
        m1, m2, c1, c2, k1, k2,
        theta_0_1, theta_0_2,
        sigma_epsilon_1, sigma_epsilon_2,
        feat_names,
        alpha1, alpha2
    )
    # Compare
    df_pruned = compare_coeffs(
        true_coeffs, initial_guess,
        expanded_names, active_feature_indices=active_idx
    )
    print(df_pruned)

    # E) Plot: True vs. Discovered Model (No Noise comparison)
    X_true_no_noise, X_dot_true_no_noise = simulate_true(
        m1, m2, c1, c2, k1, k2,
        alpha1, alpha2,
        theta_0_1, theta_0_2,
        x0, t, U,
        noise_type=None
    )
    plot_true_vs_estimated_model(
        t, X_true_no_noise, x0, U, fitted_sindy_model,
        title_prefix="In-Sample (No Noise)"
    )

    # F) Robustness Check
    x0_robust = np.array([-0.5, 0.3, 1.2, 0.0])
    X_true_robust, X_dot_true_robust = simulate_true(
        m1, m2, c1, c2, k1, k2,
        alpha1, alpha2,
        theta_0_1, theta_0_2,
        x0_robust, t, U,
        noise_type=None
    )
    plot_true_vs_estimated_model(
        t, X_true_robust, x0_robust, U, fitted_sindy_model,
        title_prefix="Robustness (New IC, No Noise)"
    )

    # G) (Optional) Bayesian Noise Estimation, MPC, etc. 
    #    Same as your existing pipeline...
