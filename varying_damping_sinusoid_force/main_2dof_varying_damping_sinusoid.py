import numpy as np
import matplotlib.pyplot as plt

# Local imports
from simulate_2dof_varying_damping_sinusoid import (
    simulate_true_varying_damping,
    compute_true_coeffs_varying_damping
)
from sindy_2dof_varying_damping_sinusoid import (
    fit_sindy_model,
    prune_sindy_features,
    compare_coeffs
)
from plot_2dof_varying_damping_sinusoid import (
    plot_true_vs_estimated_model,
    plot_mpc_results_single_control,
    plot_mpc_results_phase_single_control
)
from bayesian_noise_2dof_varying_damping_sinusoid import BayesianNoiseEstimator
from bayesian_mpc_2dof_varying_damping_sinusoid import run_bayesian_mpc_single_control


###############################################################################
# 1) System Parameters
###############################################################################
m1, m2 = 1.0, 1.0
c_total = 0.6
k1, k2 = 1.0, 1.0  # fixed stiffness
dt = 0.005
t = np.arange(0, 20, dt)
np.random.seed(42)

# alpha(t) in [alpha_min, alpha_max]
alpha_min, alpha_max = 0.2, 0.8
freq_alpha = 1  # in Hz
alpha_array = alpha_min + 0.5*(alpha_max - alpha_min)*(1.0 + np.sin(2*np.pi*freq_alpha*t))

# Forcing frequency in rad/s
forcing_freq = 1.5

forcing_amp = 1.0
sin_forcing_array = forcing_amp * np.sin(forcing_freq * t)

# Noise
noise_array_1 = np.random.normal(0, 0.1, size=len(t))
noise_array_2 = np.random.normal(0, 0.1, size=len(t))

# Initial Condition
x0 = np.array([1.0, 0.0, -0.3, 0.2])


###############################################################################
# 2) Simulate the "true" system
###############################################################################
X_true, X_dot_true = simulate_true_varying_damping(
    m1, m2,
    c_total,
    k1, k2,
    x0, t,
    alpha_array,
    noise_array_1, noise_array_2,
    forcing_freq=forcing_freq
)
print("Simulation complete. X_true shape:", X_true.shape)

###############################################################################
# 3) Fit SINDy (2 inputs => alpha(t), sin_forcing(t))
###############################################################################
fitted_model, coeffs, feat_names = fit_sindy_model(
    X_true, X_dot_true,
    alpha_array,
    sin_forcing_array,
    t,
    poly_degree=3,
    include_bias=True,
    include_interactions=True,
    stlsq_threshold=0.1,
    stlsq_alpha=0.1
)
print("\nSINDy model fitted.")
print("Feature names:", feat_names)
print("Coefficient matrix shape:", coeffs.shape)

###############################################################################
# 4) Prune & Compare to "true" expansions
###############################################################################
pruned_coeff_matrix, pruned_feat_names, active_indices = prune_sindy_features(
    fitted_model, rows_for_coeffs=(1, 3), tol=1e-5
)
print("Pruned features:", pruned_feat_names)
print("Pruned coeff matrix shape:", pruned_coeff_matrix.shape)

# Build the "true" expansions
true_v1_all, true_v2_all = compute_true_coeffs_varying_damping(
    feat_names,
    m1, m2,
    c_total,
    k1, k2,
    theta_0_1=0.0,
    theta_0_2=0.0
)
true_v1_pruned = true_v1_all[active_indices]
true_v2_pruned = true_v2_all[active_indices]

df_compare_v1 = compare_coeffs(true_v1_pruned, pruned_coeff_matrix[0], pruned_feat_names)
df_compare_v2 = compare_coeffs(true_v2_pruned, pruned_coeff_matrix[1], pruned_feat_names)

print("\nCompare v1_dot =>\n", df_compare_v1)
print("\nCompare v2_dot =>\n", df_compare_v2)

###############################################################################
# 5) Plot True vs. Estimated
###############################################################################
plot_true_vs_estimated_model(
    t, X_true, x0,
    alpha_array, sin_forcing_array,
    fitted_model,
    title_prefix="Varying Damping + Sinusoid"
)
plt.figure()
plt.plot(t, alpha_array, label='alpha(t)')
plt.plot(t, sin_forcing_array, label='sin forcing')
plt.legend()
plt.title("Inputs: alpha(t) & sinusoidal forcing")
plt.show()

###############################################################################
# 6) Bayesian Noise Estimation
###############################################################################
bayes_noise = BayesianNoiseEstimator(
    det_sindy_model=fitted_model,
    rows_for_coeffs=(1,3),
    n_walkers=16
)
bayes_noise.fit(
    X_true, X_dot_true,
    alpha_array, sin_forcing_array, t,
    n_steps=200,
    initial_sigma=(0.1,0.1)
)
sigma_samples = bayes_noise.get_sigma_samples()
print("\nPosterior samples for [sigma1, sigma2]:", sigma_samples.shape)

plt.figure()
plt.hist(sigma_samples[:,0], bins=30, alpha=0.5, label='sigma1')
plt.hist(sigma_samples[:,1], bins=30, alpha=0.5, label='sigma2')
plt.legend()
plt.title("Noise Posterior")
plt.show()

###############################################################################
# 7) Bayesian MPC (Single-Control alpha(t)) with known sinus forcing
###############################################################################
alpha_no_control = np.full(len(t), 0.5)
X_uncontrolled, _ = simulate_true_varying_damping(
    m1, m2,
    c_total,
    k1, k2,
    x0, t,
    alpha_no_control,
    noise_array_1, noise_array_2,
    forcing_freq=forcing_freq
)

N = 20
Q = np.diag([100, 1, 100, 1])
R = 0.1
x_ref = np.array([0,0,0,0])

# sample from noise posterior
n_mpc_samples = 1
idxs = np.random.choice(len(sigma_samples), size=n_mpc_samples, replace=False)
sigma_draws = sigma_samples[idxs]

from bayesian_mpc_2dof_varying_damping_sinusoid import run_bayesian_mpc_single_control

X_all, U_all, X_mean, X_std, U_mean, U_std = run_bayesian_mpc_single_control(
    pruned_feat_names,
    pruned_coeff_matrix,
    fitted_model,
    sigma_draws,
    x0, t,
    Q, R,
    N,
    alpha_min,
    alpha_max,
    x_ref,
    forcing_freq=forcing_freq,
    sin_forcing_array=sin_forcing_array
)

plot_mpc_results_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
plot_mpc_results_phase_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std)
