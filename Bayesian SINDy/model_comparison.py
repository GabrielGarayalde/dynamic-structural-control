"""
model_comparison.py

Script to:
1) Simulate the true 2DOF system
2) For multiple SINDy configurations (poly_degree, interactions),
   fit a SINDy model
3) Generate/predict each model's trajectory
4) Print a "compare_coeffs" table for each model (with pruning)
5) Plot and compare the trajectories vs. the true system
"""

import numpy as np
import matplotlib.pyplot as plt

# Adjust these imports to match your local filenames
from simulate_2dof import simulate_true, compute_true_coeffs
from sindy_2dof import (
    get_initial_guess_from_pysindy,
    build_expanded_feature_names,
    prune_sindy_features,
    compare_coeffs
)
from bayesian_mpc_2dof import run_bayesian_mpc_deterministic_noise  # if needed

#############################
# 1) SIMULATE THE TRUE SYSTEM
#############################
def simulate_true_system():
    # Hardcoded system parameters
    m1, m2 = 1.0, 1.0
    c1, c2 = 0.3, 0.3
    k1, k2 = 1.0, 1.0
    theta_0_1, theta_0_2 = 0.5, 0.5
    sigma_epsilon_1, sigma_epsilon_2 = 0.05, 0.05

    dt = 0.01
    t = np.arange(0, 15, dt)
    # np.random.seed(42)

    # Control input
    U = 0.5 * np.sin(2 * np.pi * 0.5 * t)

    # Initial conditions
    x0 = np.array([1.0, 0.0, 0.5, -0.2])

    # Noise arrays
    noise_array_1 = np.random.normal(0, sigma_epsilon_1, size=len(t))
    noise_array_2 = np.random.normal(0, sigma_epsilon_2, size=len(t))

    # Simulate "true" system
    X = simulate_true(
        m1, m2, c1, c2, k1, k2,
        theta_0_1, theta_0_2,
        x0, t, U,
        noise_array_1, noise_array_2
    )

    # Compute derivatives X_dot
    X_dot = np.zeros_like(X)
    for i in range(len(t)-1):
        x1, v1, x2, v2 = X[i]
        dx1 = v1
        dx2 = v2
        dv1 = (theta_0_1 - c1*v1 - k1*x1
               - c2*(v1 - v2) - k2*(x1 - x2))/m1 + noise_array_1[i]
        dv2 = (theta_0_2 + U[i]
               - c2*(v2 - v1) - k2*(x2 - x1))/m2 + noise_array_2[i]

        X_dot[i] = [dx1, dv1, dx2, dv2]

    # last index
    X_dot[-1] = X_dot[-2]

    # Return everything
    return t, U, X, X_dot, x0, m1, m2, c1, c2, k1, k2, theta_0_1, theta_0_2, sigma_epsilon_1, sigma_epsilon_2


#############################
# 2) PREDICT MODEL TRAJECTORY
#############################
def predict_sindy_trajectory(fitted_model, x0, t, U):
    """
    Basic Euler approach using fitted_model.predict(...) each step.
    """
    X_pred = np.zeros((len(t), len(x0)))
    X_pred[0] = x0

    dt = t[1] - t[0]
    for i in range(len(t)-1):
        x_current = X_pred[i].reshape(1,-1)
        u_current = np.array(U[i]).reshape(1, -1)
        x_dot_pred = fitted_model.predict(x_current, u=u_current)[0]
        X_pred[i+1] = X_pred[i] + dt * x_dot_pred

    return X_pred


def main():
    #############################
    # A) Simulate True System
    #############################
    (t, U, X_true, X_dot_true, x0,
     m1, m2, c1, c2, k1, k2,
     theta_0_1, theta_0_2,
     sigma_epsilon_1, sigma_epsilon_2) = simulate_true_system()

    #############################
    # B) Define SINDy Configs
    #############################
    sindy_configs = [
        # {"poly_degree":1, "include_interactions":False},
        # {"poly_degree":1, "include_interactions":True},
        # {"poly_degree":2, "include_interactions":False},
        # {"poly_degree":2, "include_interactions":True},
        {"poly_degree":3, "include_interactions":True},
        # {"poly_degree":3, "include_interactions":True},
    ]

    #############################
    # C) Loop Over Configs
    #############################
    results = {}
    for cfg in sindy_configs:
        deg = cfg["poly_degree"]
        inter = cfg["include_interactions"]

        # 1) Fit SINDy model
        initial_guess, feat_names, fitted_model = get_initial_guess_from_pysindy(
            X_true, X_dot_true, U=U, t=t,
            rows_for_coeffs=(1,3),
            poly_degree=deg,
            include_bias=True,
            include_interactions=inter
        )
        
        # Extract the coefficient matrix and feature names
        det_params_matrix = fitted_model.coefficients()
        det_feature_names = fitted_model.get_feature_names()

        # print("Deterministic parameters matrix :\n", det_params_matrix)

        # 2) Build expanded names for comparison (2*M features)
        expanded_names = build_expanded_feature_names(feat_names)

        # 3) Compute "true" deterministic coeffs for eqn
        #    (assuming we pass "feat_names" to compute_true_coeffs)
        true_coeffs = compute_true_coeffs(
            m1, m2, c1, c2, k1, k2,
            theta_0_1, theta_0_2,
            sigma_epsilon_1, sigma_epsilon_2,
            feat_names
        )

        # 4) Prune
        pruned_coeff_matrix, pruned_names, active_idx = prune_sindy_features(
            fitted_model,
            rows_for_coeffs=(1,3),
            tol=1e-6
        )
        n_pruned = pruned_coeff_matrix.shape[1]

        # 5) Compare pruned vs. true in a df
        df_pruned = compare_coeffs(
            true_coeffs, initial_guess, expanded_names,
            active_feature_indices=active_idx
        )
        # Print the table
        label_str = f"deg={deg}, inter={inter}, #pruned={n_pruned}"
        # print(f"\n=== Comparison table for {label_str} ===")
        print(df_pruned)

        # 6) Predict trajectory
        X_model = predict_sindy_trajectory(fitted_model, x0, t, U)

        # 7) MSE
        mse = np.mean((X_model - X_true)**2)

        # 8) Store results
        results[label_str] = {
            "X_model": X_model,
            "mse": mse,
            "df_pruned": df_pruned,  # we might store the table if we want
        }
        print(f"{label_str} => MSE={mse:.4e}")

    #############################
    # D) Plot All Trajectories
    #############################
    fig, axs = plt.subplots(2, 2, figsize=(12,8))
    axs = axs.flatten()
    state_labels = ['x1','v1','x2','v2']

    for i_state in range(4):
        ax = axs[i_state]
        # Plot true
        ax.plot(t, X_true[:, i_state], 'k-', label='True', linewidth=2)

        # Plot each config
        for label_str, data in results.items():
            X_m = data["X_model"]
            ax.plot(t, X_m[:, i_state], '--', label=label_str, alpha=0.7)

        ax.set_title(f"State: {state_labels[i_state]}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Value")

        if i_state == 0:  # to avoid overcluttering each panel
            ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
