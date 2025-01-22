import numpy as np
import matplotlib.pyplot as plt

# Local imports for the new "varying_damping" case
from simulate_2dof_varying_damping import (
    compute_true_coeff_v1dot, compute_true_coeff_v2dot, simulate_true_varying_damping)

from sindy_2dof_varying_damping import (
    fit_sindy_model,
    prune_sindy_features,
    compare_coeffs
)
from plot_2dof_varying_damping import (
    plot_true_vs_estimated_model,
    plot_mpc_results_single_control,
    plot_mpc_results_phase_single_control
)
from bayesian_noise_2dof_varying_damping import BayesianNoiseEstimator
from bayesian_mpc_2dof_varying_damping import run_bayesian_mpc_single_control


###############################################################################
# A) System and Simulation Parameters
###############################################################################
m1, m2 = 1.0, 1.0

# We'll keep two fixed stiffness springs, for example:
k1, k2 = 1.0, 1.0

# Now we have total damping c_total to be split by alpha(t)
c_total = 0.6  # c1(t) + c2(t) = 0.6

theta_0_1, theta_0_2 = 0.0, 0.0
sigma_epsilon_1, sigma_epsilon_2 = 0.05, 0.05

dt = 0.005
t = np.arange(0, 20, dt)
np.random.seed(42)

###############################################################################
# B) Define alpha(t) in [0,1], e.g. a sine wave in [alpha_min, alpha_max].
###############################################################################
alpha_min = 0.1
alpha_max = 0.9
freq = 0.5  # Hz
alpha_array = alpha_min + 0.5*(alpha_max - alpha_min)*(
    1 + np.sin(2*np.pi*freq*t)
)

###############################################################################
# C) Prepare initial condition and noise
###############################################################################
x0 = np.array([1.0, 0.0, 0.5, -0.2])
noise_array_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
noise_array_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))

###############################################################################
# D) Simulate "true" system
###############################################################################
X_true, X_dot_true = simulate_true_varying_damping(
    m1, m2,
    c_total,
    k1, k2,
    theta_0_1, theta_0_2,
    x0, t,
    alpha_array,
    noise_array_1, noise_array_2
)
print("Simulation complete. X_true shape:", X_true.shape)

###############################################################################
# E) Fit SINDy model
###############################################################################
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

###############################################################################
# F) Prune & Compare with "True" Coeffs
###############################################################################
pruned_coeff_matrix, pruned_feat_names, active_indices = prune_sindy_features(
    fitted_model, rows_for_coeffs=(1,3), tol=1e-5
)
print("Pruned coeff matrix shape:", pruned_coeff_matrix.shape)
print("Pruned features:", pruned_feat_names)

# For now, we'll skip the full detail. You can fill it in exactly if you want:
# After pruning:
true_v1 = [
    compute_true_coeff_v1dot(feat, m1, c_total, k1, k2, theta_0_1)
    for feat in pruned_feat_names
]
true_v2 = [
    compute_true_coeff_v2dot(feat, m2, c_total, k2, theta_0_2)
    for feat in pruned_feat_names
]

df_compare_v1 = compare_coeffs(true_v1, pruned_coeff_matrix[0], pruned_feat_names)
df_compare_v2 = compare_coeffs(true_v2, pruned_coeff_matrix[1], pruned_feat_names)
print("v1_dot comparison =>\n", df_compare_v1)
print("v2_dot comparison =>\n", df_compare_v2)

###############################################################################
# G) Minimal Plot
###############################################################################
plt.figure(figsize=(10,5))
plt.plot(t, X_true[:,0], label="x1(t)")
plt.plot(t, X_true[:,2], label="x2(t)")
plt.plot(t, alpha_array, label="alpha(t)", linestyle='--')
plt.xlabel("Time")
plt.legend()
plt.title("2DOF System with Varying Damping Ratio alpha(t)")
plt.show()

# Compare True vs. Estimated
plot_true_vs_estimated_model(
    t,
    X_true,
    x0,
    alpha_array,
    fitted_model,
    title_prefix="In-sample"
)

###############################################################################
# H) Bayesian Noise Estimation
###############################################################################
bayes_noise_model = BayesianNoiseEstimator(
    det_sindy_model=fitted_model,
    rows_for_coeffs=(1, 3),
    n_walkers=16
)
bayes_noise_model.fit(X_true, X_dot_true, U=alpha_array, t=t,
                      n_steps=200, initial_sigma=(0.1,0.1))
sigma_samples = bayes_noise_model.get_sigma_samples()
print("Posterior samples for [sigma1, sigma2]:", sigma_samples.shape)

plt.figure()
plt.hist(sigma_samples[:,0], bins=30, alpha=0.5, label='sigma_epsilon_1')
plt.hist(sigma_samples[:,1], bins=30, alpha=0.5, label='sigma_epsilon_2')
plt.legend()
plt.title("Posterior Distributions for Noise Parameters")
plt.show()

###############################################################################
# I) Single-Control MPC
###############################################################################
alpha_no_control = np.full(len(t), 0.5)
x0_mpc = np.array([-0.5, 0.3, 1.2, 0.0])

noise_array_unc_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
noise_array_unc_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))

# "Uncontrolled" reference => alpha=0.5
X_uncontrolled, _ = simulate_true_varying_damping(
    m1, m2,
    c_total,
    k1, k2,
    theta_0_1, theta_0_2,
    x0_mpc, t,
    alpha_no_control,
    noise_array_unc_1, noise_array_unc_2
)

N = 15
Q = np.diag([100, 1, 100, 1])
R = 0.1
x_ref = np.array([0,0,0,0])

n_mpc_samples = 1
idxs = np.random.choice(len(sigma_samples), size=n_mpc_samples, replace=False)
sigma_draws = sigma_samples[idxs]

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

plot_mpc_results_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
plot_mpc_results_phase_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
