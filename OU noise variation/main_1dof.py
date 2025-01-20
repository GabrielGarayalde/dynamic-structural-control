"""
main_1dof.py

Main script to:
1) Simulate a 1DOF system (the "true" model) with a single input u(t)
2) Fit a PySINDy model for the deterministic parameters (SINDy)
3) Prune SINDy features and compare them to known true coefficients
4) Plot the discovered model vs. the true system for (a) the same initial condition and
   (b) a new initial condition (robustness check)
5) Optionally, run Bayesian Noise Estimation and MPC
"""

import numpy as np
import matplotlib.pyplot as plt

# Local imports (adapted from 2DOF versions)
from simulate_1dof_linear_ou import simulate_true, compute_true_coeffs

from sindy_1dof import (
    get_initial_guess_from_pysindy,
    build_expanded_feature_names,
    compare_coeffs,
    prune_sindy_features
)

# (Optional) If you have Bayesian noise & MPC code for 1DOF, import them:
# from bayesian_noise_1dof import BayesianNoiseEstimator
# from bayesian_mpc_1dof import run_bayesian_mpc_deterministic_noise
from plot_1dof import  plot_true_vs_estimated_model


###############################################################################
# MAIN SCRIPT
###############################################################################
if __name__ == "__main__":

    # ------------------------------
    # A) System and Simulation Parameters
    # ------------------------------
    m = 1.0
    c = 0.3
    k = 1.0
    theta_0 = 0.5
    sigma_epsilon = 0.01

    dt = 0.01
    t = np.arange(0, 15, dt)

    # ------------------------------
    # B) Define Single Control u(t)
    # ------------------------------
    # Example: sinusoidal input
    u = 0.5 * np.sin(2 * np.pi * 0.5 * t)

    # Initial condition X0 = [x(0), v(0)]
    X0 = np.array([1.0, 0.0])

    # ------------------------------
    # C) Simulate the "True" 1DOF System
    # ------------------------------
    X_true, X_dot_true = simulate_true(
        m, c, k,
        theta_0,
        X0, t, u,
        mu_ou=theta_0,
        sigma_epsilon=sigma_epsilon,
        theta_ou=1.0,
        seed=42
    )

    # ------------------------------
    # D) Fit PySINDy (Deterministic Part)
    # ------------------------------
    # We only need the row for v_dot => row_for_coeffs=(1,)
    initial_guess, feat_names, fitted_sindy_model = get_initial_guess_from_pysindy(
        X_true, X_dot_true, u, t,
        row_for_coeffs=(1,),
        poly_degree=3,
        include_bias=True,
        include_interactions=True
    )

    det_params_matrix = fitted_sindy_model.coefficients()
    print("Deterministic parameters shape:\n", det_params_matrix.shape)

    # ------------------------------
    # E) Prune SINDy Features + Compare
    # ------------------------------
    pruned_coeff_vector, pruned_feature_names, active_idx = prune_sindy_features(
        fitted_sindy_model,
        row_for_coeffs=(1,),
        tol=1e-6
    )
    print("Pruned coeff vector shape:", pruned_coeff_vector.shape)

    # Build expanded names to compare with "true" coeffs
    expanded_names = build_expanded_feature_names(feat_names)
    true_coeffs = compute_true_coeffs(
        m, c, k,
        theta_0, sigma_epsilon,
        feat_names
    )

    # Show pruned comparison table
    df_pruned = compare_coeffs(true_coeffs, initial_guess, expanded_names,
                               active_feature_indices=None)
    print(df_pruned)

    # ------------------------------
    # F) (Optional) Plot: True vs. Discovered Model
    #    under same initial condition
    # ------------------------------
    plot_true_vs_estimated_model(
        t,
        X_true,  # the "true" system states
        X0,
        u,
        fitted_sindy_model,
        title_prefix="In-Sample"
    )

    # ------------------------------
    # G) (Optional) Robustness Check (New IC)
    # ------------------------------
    X0_robust = np.array([-0.5, 0.3])
    
    # If you want to test the same input 'u' but new IC:
    X_true_robust, _ = simulate_true(
        m, c, k,
        theta_0,
        X0_robust, t, u,
        mu_ou=theta_0,
        sigma_epsilon=sigma_epsilon,
        theta_ou=0.0,
        seed=46
    )
    
    plot_true_vs_estimated_model(
        t,
        X_true_robust,  # the "true" system states
        X0_robust,
        u,
        fitted_sindy_model,
        title_prefix="In-Sample"
    )
    # (Plot or compare as desired)

    # ------------------------------
    # H) (Optional) Bayesian Noise Estimation
    # ------------------------------
    # If you have a BayesianNoiseEstimator for 1DOF, you could do:
    # bayes_noise_model = BayesianNoiseEstimator(...)
    # bayes_noise_model.fit(...)
    # sigma_samples = bayes_noise_model.get_sigma_samples()
    

    # ------------------------------
    # I) (Optional) MPC
    # ------------------------------
    # If you have a 1DOF version of your MPC code:
    # X_uncontrolled = simulate_true(...)  # no control
    # ...
    # run_bayesian_mpc_deterministic_noise(...)
    # ...
    # plot_mpc_results(...)
    # etc.
