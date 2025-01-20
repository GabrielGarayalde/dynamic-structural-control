# main_2dof_varying_stiffness.py

import numpy as np
import matplotlib.pyplot as plt

# Local imports
from simulate_2dof_varying_stiffness import simulate_true_varying_stiffness
from sindy_2dof_varying_stiffness import (
    fit_sindy_model,
    prune_sindy_features,
    compare_coeffs
)

###############################################################################
# A) System and Simulation Parameters
###############################################################################
m1, m2 = 1.0, 1.0
c1, c2 = 0.3, 0.3
k_total = 2.0          # total stiffness to distribute
theta_0_1, theta_0_2 = 0.0, 0.0
sigma_epsilon_1, sigma_epsilon_2 = 0.05, 0.05

dt = 0.005
t = np.arange(0, 20, dt)
np.random.seed(42)

# B) Define alpha(t) as the "control"
# We'll pick a sine wave in [0,1]:
    # Specify alpha's bounds
alpha_min = 0.2
alpha_max = 0.8

# Frequency or angular speed for the sine wave
freq = 0.5  # Hz, for example

# Build alpha(t) to lie in [alpha_min, alpha_max]
alpha_array = alpha_min \
              + 0.5*(alpha_max - alpha_min) * (1.0 + np.sin(2 * np.pi * freq * t))

# Now alpha_array is automatically in [alpha_min, alpha_max].
# For instance, alpha_min=0.2, alpha_max=0.8 => alpha(t) in [0.2, 0.8].


# C) Prepare initial condition and noise
x0 = np.array([1.0, 0.0, 0.5, -0.2])
noise_array_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
noise_array_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))

# D) Simulate "true" system
X_true, X_dot_true = simulate_true_varying_stiffness(
    m1, m2, c1, c2,
    k_total,
    theta_0_1, theta_0_2,
    x0, t, 
    alpha_array,
    noise_array_1, noise_array_2
)

print("Simulation complete. X_true shape:", X_true.shape)

# E) Fit SINDy model
fitted_model, coeffs, feat_names = fit_sindy_model(
    X_true, X_dot_true,
    alpha_array,
    t,
    poly_degree=2,
    include_bias=True,
    include_interactions=True,
    stlsq_threshold=0.1,
    stlsq_alpha=0.1
)

print("SINDy model fitted.")
print("Coefficient matrix shape:", coeffs.shape)
print("Feature names:", feat_names)

# F) Optional: Prune & Compare
pruned_coeff_matrix, pruned_feat_names, active_indices = prune_sindy_features(
    fitted_model, rows_for_coeffs=(1,3), tol=1e-5
)
print("Pruned coeff matrix shape:", pruned_coeff_matrix.shape)
print("Pruned features:", pruned_feat_names)

# If you had a known "true" model, you can build the 'true_coeffs' array to compare
# For illustration, we'll just build a dummy vector of zeros
# or something that matches the shape for demonstration:
true_coeffs_dummy = np.zeros(pruned_coeff_matrix.shape[1])
df_compare = compare_coeffs(true_coeffs_dummy, pruned_coeff_matrix[0], pruned_feat_names)
print("\nComparison for v1_dot (row 1) =>")
print(df_compare)

df_compare_v2 = compare_coeffs(true_coeffs_dummy, pruned_coeff_matrix[1], pruned_feat_names)
print("\nComparison for v2_dot (row 3) =>")
print(df_compare_v2)

# G) Minimal Plot
plt.figure(figsize=(10,5))
plt.plot(t, X_true[:,0], label="x1(t)")
plt.plot(t, X_true[:,2], label="x2(t)")
plt.plot(t, alpha_array, label="alpha(t)", linestyle='--')
plt.xlabel("Time")
plt.legend()
plt.title("2DOF System with Varying Stiffness Ratio alpha(t)")
plt.show()

# You can add more advanced plotting or comparisons (e.g. predicted vs. true)
# by using fitted_model.simulate(...) or a custom Euler predictor.
# ...
# Suppose you have: 
#   t, X_true, x0, alpha_array, fitted_model
# where alpha_array is your control (shape (N,))
# and X_true is shape (N,4).

# Now we can plot:
from plot_2dof_varying_stiffness import plot_true_vs_estimated_model

plot_true_vs_estimated_model(
    t,
    X_true,        # "true" states from your simulator
    x0,
    alpha_array,   # single control input
    fitted_model,
    title_prefix="In-sample"
)



x0_robust = np.array([-0.5, 0.3, 1.2, 0.0])

X_true_robust, X_dot_true_robust = simulate_true_varying_stiffness(
    m1, m2, c1, c2,
    k_total,
    theta_0_1, theta_0_2,
    x0_robust, t, 
    alpha_array,
    noise_array_1, noise_array_2
)

plot_true_vs_estimated_model(
    t,
    X_true_robust,        # if you also simulated a "true" run from the new IC
    x0_robust,
    alpha_array,          # same alpha(t) or some new one
    fitted_model,
    title_prefix="Robustness Check"
)



# H) Bayesian Noise Estimation
from bayesian_noise_2dof_varying_stiffness import BayesianNoiseEstimator

bayes_noise_model = BayesianNoiseEstimator(
    det_sindy_model=fitted_model,
    rows_for_coeffs=(1,3),
    n_walkers=16
)
bayes_noise_model.fit(X_true, X_dot_true, U=alpha_array, t=t,
                      n_steps=200, initial_sigma=(0.1,0.1))
sigma_samples = bayes_noise_model.get_sigma_samples()
print("Posterior samples for [sigma1, sigma2]:", sigma_samples.shape)

# Plot noise posteriors
plt.figure()
plt.hist(sigma_samples[:,0], bins=30, alpha=0.5, label='sigma_epsilon_1')
plt.hist(sigma_samples[:,1], bins=30, alpha=0.5, label='sigma_epsilon_2')
plt.legend()
plt.title("Posterior Distributions for Noise Parameters")
plt.show()

# I) Prepare & Run Single-Control MPC
# "Uncontrolled" reference: alpha=0 => entire horizon
alpha_no_control = np.full(len(t), 0.5)  # Array of length t, all values = 0.5
# Or pick alpha_no_control=0.5,... if you prefer a different "baseline"

# We can still do a "true" simulation with that alpha_no_control
# if we want a reference trajectory for comparison
from simulate_2dof_varying_stiffness import simulate_true_varying_stiffness

x0_robust = np.array([-0.5, 0.3, 1.2, 0.0])

noise_array_unc_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
noise_array_unc_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))
X_uncontrolled, _ = simulate_true_varying_stiffness(
    m1, m2, c1, c2,
    k_total,
    theta_0_1, theta_0_2,
    x0_robust, t,
    alpha_no_control,
    noise_array_unc_1,
    noise_array_unc_2
)

# MPC parameters
N = 20
Q = np.diag([100, 1, 100, 1])
R = 0.1
x_ref = np.array([0,0,0,0])

# Sample from posterior
n_mpc_samples = 1
idxs = np.random.choice(len(sigma_samples), size=n_mpc_samples, replace=False)
sigma_draws = sigma_samples[idxs]

# Run the single-control Bayesian MPC
from bayesian_mpc_2dof_varying_stiffness import run_bayesian_mpc_single_control
X_all, U_all, X_mean, X_std, U_mean, U_std = run_bayesian_mpc_single_control(
    pruned_feat_names,
    pruned_coeff_matrix,
    fitted_model,
    sigma_draws,
    x0_robust,
    t,
    Q,
    R,
    N,
    alpha_min,
    alpha_max,
    x_ref
)

# Plot results
from plot_2dof_varying_stiffness import (
    plot_mpc_results_single_control,
    plot_mpc_results_phase_single_control
)

plot_mpc_results_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
plot_mpc_results_phase_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
