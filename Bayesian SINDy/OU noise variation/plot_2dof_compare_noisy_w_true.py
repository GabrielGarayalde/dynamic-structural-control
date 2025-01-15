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
from simulate_2dof_linear_double_control_ou import simulate_true, compute_true_coeffs

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
    theta_0_1, theta_0_2 = 0.0, 0.0
    sigma_epsilon_1, sigma_epsilon_2 = 0.1, 0.1

    dt = 0.01
    t = np.arange(0, 60, dt)

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


    # ------------------------------
    # C) Simulate the "True" 2DOF System
    # ------------------------------
    U_no_control = np.zeros((len(t), 2))
    
    
    X_noisy, X_dot_noisy = simulate_true(
        m1, m2, c1, c2, k1, k2,
        theta_0_1, theta_0_2,
        x0, t, U_no_control,
        mu_ou1=theta_0_1,
        mu_ou2=theta_0_2,
        sigma_epsilon_1=sigma_epsilon_1,
        sigma_epsilon_2=sigma_epsilon_2,
        theta_ou=1.0,       # Ornstein-Uhlenbeck parameter
        seed=None
    )
    
    
    X_no_noise, X_dot_no_noise = simulate_true(
        m1, m2, c1, c2, k1, k2,
        theta_0_1, theta_0_2,
        x0, t, U_no_control,
        mu_ou1=theta_0_1,
        mu_ou2=theta_0_2,
        sigma_epsilon_1=0,
        sigma_epsilon_2=0,
        theta_ou=1.0,       # Ornstein-Uhlenbeck parameter
        seed=48
    )
    
    
    title_prefix="No Control"
    fig, axs = plt.subplots(4, 1, figsize=(18, 16))

    # -- Top subplot: Mass 1 states
    axs[0].plot(t, X_noisy[:, 0], 'r-', label='Noisy x1')
    axs[0].plot(t, X_no_noise[:, 0], 'b-', label='No-Noise x1')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Mass 1 states')
    axs[0].set_title(f'{title_prefix}: Mass 1 (x1 & v1)')
    axs[0].legend()
    axs[0].grid(True)
    
    # -- Bottom subplot: Mass 2 states
    axs[1].plot(t, X_noisy[:, 2], 'r-', label='Noisy x2')
    axs[1].plot(t, X_no_noise[:, 2], 'b-', label='No-Noise x2')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Mass 2 states')
    axs[1].set_title(f'{title_prefix}: Mass 2 (x2 & v2)')
    axs[1].legend()
    axs[1].grid(True)
    
    axs[2].plot(t, X_noisy[:, 1], 'r--', label='Noisy v1')
    axs[2].plot(t, X_no_noise[:, 1], 'b--', label='No-Noise v1')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Mass 1 states')
    axs[2].set_title(f'{title_prefix}: Mass 1 (x1 & v1)')
    axs[2].legend()
    axs[2].grid(True)
    
    # -- Bottom subplot: Mass 2 states
    axs[3].plot(t, X_noisy[:, 3], 'r--', label='Noisy v2')
    axs[3].plot(t, X_no_noise[:, 3], 'b--', label='No-Noise v2')
    axs[3].set_xlabel('Time [s]')
    axs[3].set_ylabel('Mass 2 states')
    axs[3].set_title(f'{title_prefix}: Mass 2 (x2 & v2)')
    axs[3].legend()
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.show()


