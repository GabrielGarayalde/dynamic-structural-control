"TwoDOF plotting functions"

import matplotlib.pyplot as plt
import numpy as np
import corner
# from TwoDOF_feature_definitions import get_feature_names

# ------------------------------
# Parameter Diagnostics and Comparison
# ------------------------------

def plot_true_vs_estimated_uncontrolled_for_ic(t, X_true_uncontrolled, X_estimated_uncontrolled_simulations, initial_condition_label="IC"):
    """
    Plot the system behavior under u=0 using True Parameters vs Estimated Parameters with Uncertainty.
    Now we have four states: x1, v1, x2, v2.
    """
    pred_mean = np.mean(X_estimated_uncontrolled_simulations, axis=0)
    pred_std = np.std(X_estimated_uncontrolled_simulations, axis=0)
    states = ['x1', 'v1', 'x2', 'v2']
    fig, axs = plt.subplots(4,1,figsize=(10,12))
    for i in range(4):
        axs[i].plot(t, X_true_uncontrolled[:,i], 'b-', label='True Params (u=0)')
        axs[i].plot(t, pred_mean[:,i], 'r--', label='Estimated Params Mean (u=0)')
        axs[i].fill_between(t, pred_mean[:,i]-2*pred_std[:,i], pred_mean[:,i]+2*pred_std[:,i],
                            color='r', alpha=0.2, label='95% CI')
        axs[i].set_xlabel('Time [s]')
        axs[i].set_ylabel(states[i])
        axs[i].legend()
        axs[i].grid(True)
        axs[i].set_title(f'{initial_condition_label}: {states[i]} under u=0: True vs Estimated')
    plt.tight_layout()
    plt.show()

def plot_parameter_distributions(model):
    """
    Plot the posterior distributions of the identified parameters using corner plots.
    For the 2DOF system, assume parameter vector structure:
    [theta_0_1, coeffs_1..., theta_0_2, coeffs_2..., sigma_epsilon_1, sigma_epsilon_2]

    If get_feature_names() returns M features, we have:
    - 1 parameter for theta_0_1
    - M parameters for coeffs_1
    - 1 parameter for theta_0_2
    - M parameters for coeffs_2
    - 2 parameters for sigma_epsilon_1 and sigma_epsilon_2

    Total = 2*(1+M)+2
    """

    corner.corner(model.samples, labels=model.expanded_names,
                  quantiles=[0.16,0.5,0.84], show_titles=True, title_kwargs={"fontsize":12})
    plt.show()

def print_parameter_comparison(true_values, model, feature_names):
    """
    Compare true coefficients with estimated parameters and report errors and uncertainties.
    Assuming parameter vector:
    [theta_0_1, coeffs_1..., theta_0_2, coeffs_2..., sigma_epsilon_1, sigma_epsilon_2]

    We'll align with what we defined in plot_parameter_distributions.
    """


    print("\n### Parameter Comparison ###\n")

    def format_comparison(tv, est, lbl):
        median = np.median(est)
        std = np.std(est)
        if tv == 0:
            err = np.abs(median)
            unc = std
            return f"{lbl:25}: True={tv:.5f}, Est={median:.5f}±{std:.5f}, AbsErr={err:.5f}, Unc={unc:.5f}"
        else:
            perr = 100 * np.abs(median - tv) / np.abs(tv)
            unc = 100 * std / np.abs(tv)
            return f"{lbl:25}: True={tv:.5f}, Est={median:.5f}±{std:.5f}, Err={perr:.2f}%, Unc={unc:.2f}%"

    if len(true_values) != len(feature_names):
        print("Warning: Number of true values does not match number of parameters.")
        min_len = min(len(true_values), len(feature_names))
        true_values = true_values[:min_len]
        feature_names = feature_names[:min_len]

    for i, lbl in enumerate(feature_names):
        tv = true_values[i]
        est = model.samples[:, i]
        print(format_comparison(tv, est, lbl))

# ------------------------------
# Plotting Functions for MPC and Comparisons
# ------------------------------

def plot_uncontrolled_vs_mpc(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot the Uncontrolled Trajectory vs MPC Mean Trajectory with uncertainty bounds.
    States: x1, v1, x2, v2
    """
    states = ['x1', 'v1', 'x2', 'v2']
    fig, axs = plt.subplots(3, 1, figsize=(14, 12)) 
    # We will plot states in the first 2 axes combined and control in the 3rd

    # Plot States
    # For clarity, plot x1, v1 on the first subplot and x2, v2 on the second
    axs[0].plot(t, X_uncontrolled[:, 0], 'g-', label='Uncontrolled x1')
    axs[0].plot(t, X_uncontrolled[:, 1], 'g--', label='Uncontrolled v1')
    axs[0].plot(t, X_mean[:, 0], 'r-', label='MPC Mean x1')
    axs[0].plot(t, X_mean[:, 1], 'r--', label='MPC Mean v1')
    axs[0].fill_between(t, X_mean[:, 0]-2*X_std[:, 0], X_mean[:, 0]+2*X_std[:, 0],
                        color='r', alpha=0.2, label='95% CI x1')
    axs[0].fill_between(t, X_mean[:, 1]-2*X_std[:, 1], X_mean[:, 1]+2*X_std[:, 1],
                        color='r', alpha=0.2, label='95% CI v1')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('States (Mass 1)')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title('State Comparison (Mass 1): Uncontrolled vs MPC Controlled')

    axs[1].plot(t, X_uncontrolled[:, 2], 'g-', label='Uncontrolled x2')
    axs[1].plot(t, X_uncontrolled[:, 3], 'g--', label='Uncontrolled v2')
    axs[1].plot(t, X_mean[:, 2], 'r-', label='MPC Mean x2')
    axs[1].plot(t, X_mean[:, 3], 'r--', label='MPC Mean v2')
    axs[1].fill_between(t, X_mean[:, 2]-2*X_std[:, 2], X_mean[:, 2]+2*X_std[:, 2],
                        color='r', alpha=0.2, label='95% CI x2')
    axs[1].fill_between(t, X_mean[:, 3]-2*X_std[:, 3], X_mean[:, 3]+2*X_std[:, 3],
                        color='r', alpha=0.2, label='95% CI v2')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('States (Mass 2)')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title('State Comparison (Mass 2): Uncontrolled vs MPC Controlled')

    # Plot Control Inputs
    axs[2].plot(t[:-1], U_mean, 'b-', label='MPC Mean Control Input (on Mass 2)')
    axs[2].fill_between(t[:-1], U_mean - 2*U_std, U_mean + 2*U_std,
                       color='b', alpha=0.2, label='95% CI')
    axs[2].axhline(0, color='k', linestyle='--', label='Uncontrolled u=0')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Control Input u')
    axs[2].set_title('MPC Control Input with Uncertainty Bounds')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

def plot_mpc_results(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot the Uncontrolled Trajectory and MPC Mean Trajectory with uncertainty bounds for 2DOF.
    """
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))
    states = ['x1', 'v1', 'x2', 'v2']

    # Mass 1 states
    axs[0].plot(t, X_uncontrolled[:, 0], 'g-', label='Uncontrolled x1')
    axs[0].plot(t, X_uncontrolled[:, 1], 'g--', label='Uncontrolled v1')
    axs[0].plot(t, X_mean[:, 0], 'r-', label='MPC Mean x1')
    axs[0].plot(t, X_mean[:, 1], 'r--', label='MPC Mean v1')
    axs[0].fill_between(t, X_mean[:, 0]-2*X_std[:, 0], X_mean[:, 0]+2*X_std[:, 0],
                        color='r', alpha=0.2, label='95% CI x1')
    axs[0].fill_between(t, X_mean[:, 1]-2*X_std[:, 1], X_mean[:, 1]+2*X_std[:, 1],
                        color='r', alpha=0.2, label='95% CI v1')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('States (Mass 1)')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title('State Comparison (Mass 1): Uncontrolled vs MPC Controlled')

    # Mass 2 states
    axs[1].plot(t, X_uncontrolled[:, 2], 'g-', label='Uncontrolled x2')
    axs[1].plot(t, X_uncontrolled[:, 3], 'g--', label='Uncontrolled v2')
    axs[1].plot(t, X_mean[:, 2], 'r-', label='MPC Mean x2')
    axs[1].plot(t, X_mean[:, 3], 'r--', label='MPC Mean v2')
    axs[1].fill_between(t, X_mean[:, 2]-2*X_std[:, 2], X_mean[:, 2]+2*X_std[:, 2],
                        color='r', alpha=0.2, label='95% CI x2')
    axs[1].fill_between(t, X_mean[:, 3]-2*X_std[:, 3], X_mean[:, 3]+2*X_std[:, 3],
                        color='r', alpha=0.2, label='95% CI v2')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('States (Mass 2)')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title('State Comparison (Mass 2): Uncontrolled vs MPC Controlled')

    # Control Inputs
    axs[2].plot(t[:-1], U_mean, 'b-', label='MPC Mean Control Input (on Mass 2)')
    axs[2].fill_between(t[:-1], U_mean - 2*U_std, U_mean + 2*U_std,
                       color='b', alpha=0.2, label='95% CI')
    axs[2].axhline(0, color='k', linestyle='--', label='Uncontrolled u=0')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Control Input u')
    axs[2].set_title('MPC Control Input with Uncertainty Bounds')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
