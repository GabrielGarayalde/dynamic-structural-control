"""
main_2dof.py

Example main script that uses Brownian or Ornstein-Uhlenbeck noise
instead of discrete i.i.d. Gaussian for the 2DOF linear system.
"""

import numpy as np
import matplotlib.pyplot as plt

from simulate_2dof_linear_ou import (
    simulate_true,
    compute_true_coeffs
)

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

    # A) System and Simulation Parameters
    m1, m2 = 1.0, 1.0
    k1, k2 = 1.0, 1.0
    c1, c2 = 0.3, 0.3
    theta_0_1, theta_0_2 = 0.5, 0.5
    sigma_epsilon_1, sigma_epsilon_2 = 0.1, 0.1

    dt = 0.005
    t = np.arange(0, 20, dt)

    # Control input on second mass
    U = 0.5 * np.sin(2*np.pi*0.5*t)

    # Initial conditions
    x0 = np.array([1.0, 0.0, 0.5, -0.2])

    # -----------------------
    # Pick your noise type:
    #  - 'brownian' or 'ou' or None
    # -----------------------
    # noise_type = 'brownian'
    noise_type = 'ou'

    # B) Simulate the "True" 2DOF System with chosen noise
    # For OU, we can pass in some parameters
    # e.g. a mild mean-reversion rate
    X_true, X_dot_true = simulate_true(
        m1, m2, c1, c2, k1, k2,
        theta_0_1, theta_0_2,
        x0, t, U,
        sigma_epsilon_1=sigma_epsilon_1,
        sigma_epsilon_2=sigma_epsilon_2,
        theta_ou=1.0,       # Ornstein-Uhlenbeck parameter
        mu_ou1=theta_0_1,
        mu_ou2=theta_0_2,
        seed=42
    )


    # C) Fit SINDy (Deterministic Part)
    initial_guess, feat_names, fitted_sindy_model = get_initial_guess_from_pysindy(
        X_true, X_dot_true, U, t,
        rows_for_coeffs=(1, 3),
        poly_degree=3,
        include_bias=True,
        include_interactions=True
    )

    det_params_matrix = fitted_sindy_model.coefficients()
    det_feature_names = fitted_sindy_model.get_feature_names()
    print("Deterministic parameters matrix shape:", det_params_matrix.shape)

    # D) Prune SINDy + Compare
    pruned_coeff_matrix, pruned_feature_names, active_idx = prune_sindy_features(
        fitted_sindy_model, rows_for_coeffs=(1,3), tol=1e-6
    )
    expanded_names = build_expanded_feature_names(feat_names)
    print("Pruned coeff matrix shape:", pruned_coeff_matrix.shape)

    true_coeffs = compute_true_coeffs(
        m1, m2, c1, c2, k1, k2,
        theta_0_1, theta_0_2,
        sigma_epsilon_1, sigma_epsilon_2,
        feat_names
    )
    df_pruned = compare_coeffs(true_coeffs, initial_guess, expanded_names, active_feature_indices=active_idx)
    print(df_pruned)

    # E) Plot: True vs. Discovered Model (No Noise comparison)
    X_true_no_noise, X_dot_true_no_noise = simulate_true(
        m1, m2, c1, c2, k1, k2,
        theta_0_1, theta_0_2,
        x0, t, U,
        noise_type=None  # zero noise
    )
    plot_true_vs_estimated_model(
        t, X_true_no_noise, x0, U, fitted_sindy_model,
        title_prefix="In-Sample (No Noise)"
    )

    # F) Robustness Check
    x0_robust = np.array([-0.5, 0.3, 1.2, 0.0])
    X_true_robust, X_dot_true_robust = simulate_true(
        m1, m2, c1, c2, k1, k2,
        theta_0_1, theta_0_2,
        x0_robust, t, U,
        noise_type=None
    )
    plot_true_vs_estimated_model(
        t, X_true_robust, x0_robust, U, fitted_sindy_model,
        title_prefix="Robustness (New IC, No Noise)"
    )

    # G) Bayesian Noise, MPC, etc. (unchanged)...

