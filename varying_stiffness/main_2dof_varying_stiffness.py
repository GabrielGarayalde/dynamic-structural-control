import numpy as np
import matplotlib.pyplot as plt

# Local imports
from simulate_2dof_varying_stiffness import (
    simulate_true_varying_stiffness, true_coefficient_v1dot, true_coefficient_v2dot)
from sindy_2dof_varying_stiffness import (
    fit_sindy_model,
    prune_sindy_features,
    compare_coeffs
)
from plot_2dof_varying_stiffness import plot_true_vs_estimated_model
from bayesian_noise_2dof_varying_stiffness import BayesianNoiseEstimator
from bayesian_mpc_2dof_varying_stiffness import run_bayesian_mpc_single_control
from plot_2dof_varying_stiffness import (
    plot_mpc_results_single_control,
    plot_mpc_results_phase_single_control
)


###############################################################################
# I. DEFINE SYSTEM AND SIMULATION PARAMETERS
###############################################################################
# Masses
m1, m2 = 1.0, 1.0

# Damping
c1, c2 = 0.3, 0.3

# Stiffness
k_total = 2.0  # total stiffness to be distributed: k1(t) + k2(t) = k_total

# Constant "forcing" terms (could be zero or nonzero)
theta_0_1, theta_0_2 = 0.5, 0.5

# Noise levels for v1_dot and v2_dot
sigma_epsilon_1, sigma_epsilon_2 = 0.1, 0.1

# Time grid
dt = 0.005
t = np.arange(0, 20, dt)

# Random seed for reproducibility
np.random.seed(42)

###############################################################################
# II. DEFINE ALPHA(t) "CONTROL" IN [alpha_min, alpha_max]
###############################################################################
alpha_min = 0.2
alpha_max = 0.8
freq = 0.5  # frequency in Hz for alpha(t)

# Build alpha(t) in [0.2, 0.8], e.g. alpha(t) = 0.2 + 0.5*(0.8-0.2)*(1 + sin(...))
alpha_array = alpha_min + 0.5 * (alpha_max - alpha_min) * (
    1.0 + np.sin(2 * np.pi * freq * t)
)

###############################################################################
# III. PREPARE INITIAL CONDITIONS AND NOISE TRAJECTORIES
###############################################################################
x0 = np.array([1.0, 0.0, 0.5, -0.2])

# White noise arrays for v1_dot and v2_dot
noise_array_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
noise_array_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))

###############################################################################
# IV. SIMULATE "TRUE" SYSTEM TRAJECTORY
###############################################################################
X_true, X_dot_true = simulate_true_varying_stiffness(
    m1, m2, c1, c2,
    k_total,
    theta_0_1, theta_0_2,
    x0, t,
    alpha_array,
    noise_array_1, noise_array_2
)
print("Simulation complete. X_true shape:", X_true.shape)

###############################################################################
# V. FIT SINDY MODEL WITH ALPHA(t) AS A (SINGLE) CONTROL INPUT
###############################################################################
fitted_model, coeffs, feat_names = fit_sindy_model(
    X_true, X_dot_true,
    alpha_array,
    t,
    poly_degree=3,
    include_bias=True,
    include_interactions=True,
    stlsq_threshold=0.1,
    stlsq_alpha=0.1
)
print("\nSINDy model fitted.")
print("Coefficient matrix shape:", coeffs.shape)
print("Feature names:", feat_names)

###############################################################################
# VI. PRUNE FEATURES AND BUILD A "TRUE" COEFFICIENT VECTOR
#     FOR COMPARISON WITH THE ESTIMATED COEFFICIENTS
###############################################################################
pruned_coeff_matrix, pruned_feat_names, active_indices = prune_sindy_features(
    fitted_model, rows_for_coeffs=(1, 3), tol=1e-5
)
print("\nPruned coefficient matrix shape:", pruned_coeff_matrix.shape)
print("Pruned features:", pruned_feat_names)

# Build the "true" coefficients for each row (v1_dot, v2_dot) 
# by parsing the pruned_feat_names.  Remember our states are 
# X = [x1, v1, x2, v2], i.e. PySINDy calls them x0,x1,x2,x3, 
# and the single control alpha => u0.

# Build the “true” coefficient vectors for the pruned features
# For v1_dot (row=0 in pruned_coeff_matrix):
true_coeffs_v1 = [
    true_coefficient_v1dot(f, m1, c1, c2, k_total, theta_0_1)
    for f in pruned_feat_names
]
# For v2_dot (row=1 in pruned_coeff_matrix):
true_coeffs_v2 = [
    true_coefficient_v2dot(f, m2, c1, c2, k_total, theta_0_2)
    for f in pruned_feat_names
]

# Compare each row
df_compare_v1 = compare_coeffs(true_coeffs_v1, pruned_coeff_matrix[0], pruned_feat_names)
df_compare_v2 = compare_coeffs(true_coeffs_v2, pruned_coeff_matrix[1], pruned_feat_names)

print("\nComparison for v1_dot (row 1):")
print(df_compare_v1)
print("\nComparison for v2_dot (row 3):")
print(df_compare_v2)

###############################################################################
# VII. MINIMAL PLOT OF TRUE STATES & ALPHA
###############################################################################
plt.figure(figsize=(10, 5))
plt.plot(t, X_true[:, 0], label="x1(t)")
plt.plot(t, X_true[:, 2], label="x2(t)")
plt.plot(t, alpha_array, label="alpha(t)", linestyle='--')
plt.xlabel("Time")
plt.legend()
plt.title("2DOF System with Varying Stiffness Ratio alpha(t)")
plt.show()

###############################################################################
# VIII. PLOT TRUE VS. ESTIMATED TRAJECTORIES (IN-SAMPLE AND ROBUSTNESS CHECK)
###############################################################################
plot_true_vs_estimated_model(
    t,
    X_true,  # "true" states
    x0,
    alpha_array,
    fitted_model,
    title_prefix="In-sample"
)

# A second initial condition for a "robustness" check:
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
    X_true_robust,
    x0_robust,
    alpha_array,
    fitted_model,
    title_prefix="Robustness Check"
)

###############################################################################
# IX. BAYESIAN NOISE ESTIMATION
###############################################################################
bayes_noise_model = BayesianNoiseEstimator(
    det_sindy_model=fitted_model,
    rows_for_coeffs=(1, 3),
    n_walkers=16
)
bayes_noise_model.fit(
    X_true, X_dot_true,
    U=alpha_array,
    t=t,
    n_steps=200,
    initial_sigma=(0.1, 0.1)
)
sigma_samples = bayes_noise_model.get_sigma_samples()
print("\nPosterior samples for [sigma1, sigma2]:", sigma_samples.shape)

plt.figure()
plt.hist(sigma_samples[:, 0], bins=30, alpha=0.5, label='sigma_epsilon_1')
plt.hist(sigma_samples[:, 1], bins=30, alpha=0.5, label='sigma_epsilon_2')
plt.legend()
plt.title("Posterior Distributions for Noise Parameters")
plt.show()

###############################################################################
# X. SINGLE-CONTROL (alpha) MPC EXAMPLE
###############################################################################
alpha_no_control = np.full(len(t), 0.5)  # baseline alpha(t)=0.5 for entire horizon

x0_mpc = np.array([-0.5, 0.3, 1.2, 0.0])
noise_array_unc_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
noise_array_unc_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))

# Uncontrolled / baseline reference simulation
X_uncontrolled, _ = simulate_true_varying_stiffness(
    m1, m2, c1, c2,
    k_total,
    theta_0_1, theta_0_2,
    x0_mpc, t,
    alpha_no_control,
    noise_array_unc_1,
    noise_array_unc_2
)

# MPC horizon and cost parameters
N = 20
Q = np.diag([100, 1, 100, 1])
R = 0.1
x_ref = np.array([0, 0, 0, 0])

# Sample from posterior
n_mpc_samples = 1
idxs = np.random.choice(len(sigma_samples), size=n_mpc_samples, replace=False)
sigma_draws = sigma_samples[idxs]

# Run Single-Control Bayesian MPC
from sindy_2dof_varying_stiffness import prune_sindy_features  # (already imported above)
from bayesian_mpc_2dof_varying_stiffness import run_bayesian_mpc_single_control

# pruned_coeff_matrix (2, #features) => row 0 => v1_dot, row 1 => v2_dot
X_all, U_all, X_mean, X_std, U_mean, U_std = run_bayesian_mpc_single_control(
    pruned_feat_names,
    pruned_coeff_matrix,
    fitted_model,
    sigma_draws,
    x0_mpc,
    t,
    Q,
    R,
    N,
    alpha_min,
    alpha_max,
    x_ref
)

# Plot results
plot_mpc_results_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
plot_mpc_results_phase_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
