"""
main_2dof.py

Main script to:
1) Simulate a 2DOF system
2) Fit a PySINDy model for the deterministic parameters
3) Estimate noise parameters with a Bayesian approach
4) Perform MPC using the deterministic + noise parameters
5) Plot results
"""

import numpy as np
import matplotlib.pyplot as plt

from simulate_2dof import (
    simulate_true,compute_true_coeffs
)
from sindy_2dof import (
    get_initial_guess_from_pysindy,
    build_expanded_feature_names,
    compare_coeffs
)
from bayesian_noise_2dof import BayesianNoiseEstimator
from bayesian_mpc_2dof import run_bayesian_mpc_deterministic_noise

from plot_2dof import plot_mpc_results  # or your own plotting module

# ------------------------------
# System and Simulation Parameters
# ------------------------------
m1, m2 = 1.0, 1.0
k1, k2 = 1.0, 1.0
c1, c2 = 0.3, 0.3
theta_0_1, theta_0_2 = 0.5, 0.5
sigma_epsilon_1, sigma_epsilon_2 = 0.1, 0.1

dt = 0.01
t = np.arange(0, 15, dt)
np.random.seed(42)

# Control input (u) on the second mass
U = 0.5 * np.sin(2 * np.pi * 0.5 * t)

# Initial conditions for the simulation
x0 = np.array([1.0, 0.0, 0.5, -0.2])

# Noise arrays
noise_array_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
noise_array_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))

# ------------------------------
# A) Simulate the "True" 2DOF System
# ------------------------------
X = simulate_true(
    m1, m2, c1, c2, k1, k2, 
    theta_0_1, theta_0_2, 
    x0, t, U, 
    noise_array_1, noise_array_2
)

# Compute derivatives X_dot from the simulated trajectory
X_dot = np.zeros_like(X)
for i in range(len(t) - 1):
    x1, v1, x2, v2 = X[i]
    dx1 = v1
    dx2 = v2
    dv1 = (theta_0_1 - c1*v1 - k1*x1 - c2*(v1 - v2) - k2*(x1 - x2))/m1 + noise_array_1[i]
    dv2 = (theta_0_2 + U[i] - c2*(v2 - v1) - k2*(x2 - x1))/m2 + noise_array_2[i]
    X_dot[i] = [dx1, dv1, dx2, dv2]
X_dot[-1] = X_dot[-2]

# ------------------------------
# B) Fit PySINDy (Deterministic Part)
# ------------------------------
initial_guess, feat_names, fitted_sindy_model = get_initial_guess_from_pysindy(
    X, X_dot, U, t,
    rows_for_coeffs=(1, 3),
    sigma_epsilon_1=0.1, 
    sigma_epsilon_2=0.1,
    poly_degree=2,
    include_bias=True,
    include_interactions=False
)

# Extract the coefficient matrix and feature names
det_params_matrix = fitted_sindy_model.coefficients()
det_feature_names = fitted_sindy_model.get_feature_names()

print("Deterministic parameters (SINDy) discovered:\n", det_params_matrix)
print("Deterministic feature names:\n", det_feature_names)




" PRUNING FEATURES"
tol = 1e-6
row_v1dot = det_params_matrix[1, :]
row_v2dot = det_params_matrix[3, :]

nonzero_idx_v1 = np.where(np.abs(row_v1dot) > tol)[0]
nonzero_idx_v2 = np.where(np.abs(row_v2dot) > tol)[0]

# Union of indices used in either row
active_features = np.union1d(nonzero_idx_v1, nonzero_idx_v2)
print("Original # features:", det_params_matrix.shape[1])
print("Pruned # features:", len(active_features))

original_names = fitted_sindy_model.get_feature_names()
pruned_feature_names = [original_names[i] for i in active_features]

row_v1dot_pruned = row_v1dot[active_features]  # shape (len(active_features),)
row_v2dot_pruned = row_v2dot[active_features]

# Create a pruned 2 x (n_pruned_features) matrix
# where row 0 => v1_dot, row 1 => v2_dot
pruned_coeff_matrix = np.vstack([row_v1dot_pruned, row_v2dot_pruned])

print("pruned_coeff_matrix shape:", pruned_coeff_matrix.shape)
print("pruned_feature_names:", pruned_feature_names)



# Compare to ground-truth
expanded_names = build_expanded_feature_names(feat_names)
true_coeffs = compute_true_coeffs(
    m1, m2, c1, c2, k1, k2, 
    theta_0_1, theta_0_2, 
    sigma_epsilon_1, sigma_epsilon_2,
    feat_names
)
comparison_df = compare_coeffs(true_coeffs, initial_guess, expanded_names)
print("\nComparison with True Coeffs:\n", comparison_df)




# ------------------------------
# C) Bayesian Noise Estimation
# ------------------------------
bayes_noise_model = BayesianNoiseEstimator(
    det_sindy_model=fitted_sindy_model,
    rows_for_coeffs=(1, 3),
    n_walkers=20
)
bayes_noise_model.fit(X, X_dot, U=U, t=t, n_steps=200, initial_sigma=(0.0, 0.0))
sigma_samples = bayes_noise_model.get_sigma_samples()
print("Posterior samples for [sigma1, sigma2]:", sigma_samples.shape)

# Optionally, plot the noise posterior
plt.figure()
plt.hist(sigma_samples[:,0], bins=30, alpha=0.5, label='sigma_epsilon_1')
plt.hist(sigma_samples[:,1], bins=30, alpha=0.5, label='sigma_epsilon_2')
plt.legend()
plt.title("Posterior Distributions for Noise Parameters")
plt.show()

# ------------------------------
# D) Prepare for MPC
# ------------------------------
# 1) Uncontrolled system for comparison
noise_array_unc_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
noise_array_unc_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))
U_no_control = np.zeros(len(t))

X_uncontrolled = simulate_true(
    m1, m2, c1, c2, k1, k2, 
    theta_0_1, theta_0_2, 
    x0, t, U_no_control, 
    noise_array_unc_1, 
    noise_array_unc_2
)

X_dot_uncontrolled = np.zeros_like(X_uncontrolled)
for i in range(len(t) - 1):
    x1, v1, x2, v2 = X_uncontrolled[i]
    dx1 = v1
    dx2 = v2
    dv1 = (theta_0_1 - c1*v1 - k1*x1 - c2*(v1 - v2) - k2*(x1 - x2))/m1 + noise_array_unc_1[i]
    dv2 = (theta_0_2 + 0 - c2*(v2 - v1) - k2*(x2 - x1))/m2 + noise_array_unc_2[i]
    X_dot_uncontrolled[i] = [dx1, dv1, dx2, dv2]
X_dot_uncontrolled[-1] = X_dot_uncontrolled[-2]

# 2) MPC parameters
N = 15
Q = np.diag([100, 1, 100, 1])
R = 0.1
u_max, u_min = 1.0, -1.0
x_ref = np.array([0.0, 0.0, 0.0, 0.0])

# 3) Choose some noise samples from posterior
n_mpc_samples = 1
idxs = np.random.choice(len(sigma_samples), size=n_mpc_samples, replace=False)
sigma_draws = sigma_samples[idxs]

# ------------------------------
# E) Run MPC
# ------------------------------
X_all, U_all, X_mean, X_std, U_mean, U_std = run_bayesian_mpc_deterministic_noise(
    fitted_sindy_model, 
    sigma_draws, 
    x0, t, Q, R, N, u_min, u_max, x_ref
)

# ------------------------------
# F) Plot MPC vs Uncontrolled
# ------------------------------
plot_mpc_results(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
