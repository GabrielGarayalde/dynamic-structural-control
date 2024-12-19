import matplotlib.pyplot as plt
import numpy as np
import corner
from SDOF_feature_definitions import get_feature_names
# ------------------------------
# 3. Parameter Diagnostics and Comparison
# ------------------------------

def plot_true_vs_estimated_uncontrolled_for_ic(t, X_true_uncontrolled, X_estimated_uncontrolled_simulations, initial_condition_label="IC"):
    """
    Plot the system behavior under u=0 using True Parameters vs Estimated Parameters with Uncertainty.
    """
    pred_mean = np.mean(X_estimated_uncontrolled_simulations, axis=0)
    pred_std = np.std(X_estimated_uncontrolled_simulations, axis=0)
    states = ['x', 'v']
    fig, axs = plt.subplots(2,1,figsize=(10,8))
    for i in range(2):
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
    """
    labels = ['theta_0'] + [f'c_v_{feat}' for feat in get_feature_names()] + ['sigma_epsilon']
    corner.corner(model.samples, labels=labels,
                  quantiles=[0.16,0.5,0.84], show_titles=True, title_kwargs={"fontsize":12})
    plt.show()

def print_parameter_comparison(true_values, model, feature_names):
    """
    Compare true coefficients with estimated parameters and report errors and uncertainties.
    """
    print("\n### Parameter Comparison ###\n")

    param_labels = ['theta_0', 
                    'c_v_x', 'c_v_v', 
                    'c_v_x²', 'c_v_v²', 
                    'c_v_x*v',
                    'c_v_u', 
                    'sigma_epsilon']

    def format_comparison(tv, est, lbl):
        median = np.median(est)
        std = np.std(est)
        # if lbl == 'sigma_epsilon':
        #     abs_err = np.abs(median - tv)
        #     return f"{lbl:20}: True={tv:.5f}, Est={median:.5f}±{std:.5f}, AbsErr={abs_err:.5f}"
        if tv == 0:
            err = np.abs(median)
            unc = std
            return f"{lbl:20}: True={tv:.5f}, Est={median:.5f}±{std:.5f}, AbsErr={err:.5f}, Unc={unc:.5f}"
        else:
            perr = 100 * np.abs(median - tv) / np.abs(tv)
            unc = 100 * std / np.abs(tv)
            return f"{lbl:20}: True={tv:.5f}, Est={median:.5f}±{std:.5f}, Err={perr:.2f}%, Unc={unc:.2f}%"

    if len(true_values) != len(param_labels):
        print("Warning: Number of true values does not match number of parameters.")
        min_len = min(len(true_values), len(param_labels))
        true_values = true_values[:min_len]
        param_labels = param_labels[:min_len]

    for i, lbl in enumerate(param_labels):
        tv = true_values[i]
        est = model.samples[:, i]
        print(format_comparison(tv, est, lbl))
        
        
# ------------------------------
# 6. Plotting Functions
# ------------------------------

def plot_uncontrolled_vs_mpc(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot the Uncontrolled Trajectory vs MPC Mean Trajectory with uncertainty bounds.
    Also plot the MPC control input vs. uncontrolled (u=0).
    """
    states = ['x', 'v']
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))

    # Plot States
    axs[0].plot(t, X_uncontrolled[:, 0], 'g-', label='Uncontrolled x')
    axs[0].plot(t, X_uncontrolled[:, 1], 'g--', label='Uncontrolled v')
    axs[0].plot(t, X_mean[:, 0], 'r-', label='MPC Mean x')
    axs[0].plot(t, X_mean[:, 1], 'r--', label='MPC Mean v')
    axs[0].fill_between(t, X_mean[:, 0]-2*X_std[:, 0], X_mean[:, 0]+2*X_std[:, 0],
                        color='r', alpha=0.2, label='95% CI x')
    axs[0].fill_between(t, X_mean[:, 1]-2*X_std[:, 1], X_mean[:, 1]+2*X_std[:, 1],
                        color='r', alpha=0.2, label='95% CI v')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('States')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title('State Comparison: Uncontrolled vs MPC Controlled')

    # Plot Control Inputs
    axs[1].plot(t[:-1], U_mean, 'b-', label='MPC Mean Control Input')
    axs[1].fill_between(t[:-1], U_mean - 2*U_std, U_mean + 2*U_std,
                       color='b', alpha=0.2, label='95% CI')
    axs[1].axhline(0, color='k', linestyle='--', label='Uncontrolled u=0')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Control Input u')
    axs[1].set_title('MPC Control Input with Uncertainty Bounds')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_mpc_results(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot the Uncontrolled Trajectory and MPC Mean Trajectory with uncertainty bounds.
    """
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))
    states = ['x', 'v']

    # Plot States
    axs[0].plot(t, X_uncontrolled[:, 0], 'g-', label='Uncontrolled x')
    axs[0].plot(t, X_uncontrolled[:, 1], 'g--', label='Uncontrolled v')
    axs[0].plot(t, X_mean[:, 0], 'r-', label='MPC Mean x')
    axs[0].plot(t, X_mean[:, 1], 'r--', label='MPC Mean v')
    axs[0].fill_between(t, X_mean[:, 0]-2*X_std[:, 0], X_mean[:, 0]+2*X_std[:, 0],
                        color='r', alpha=0.2, label='95% CI x')
    axs[0].fill_between(t, X_mean[:, 1]-2*X_std[:, 1], X_mean[:, 1]+2*X_std[:, 1],
                        color='r', alpha=0.2, label='95% CI v')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('States')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title('State Comparison: Uncontrolled vs MPC Controlled')

    # Plot Control Inputs
    axs[1].plot(t[:-1], U_mean, 'b-', label='MPC Mean Control Input')
    axs[1].fill_between(t[:-1], U_mean - 2*U_std, U_mean + 2*U_std,
                       color='b', alpha=0.2, label='95% CI')
    axs[1].axhline(0, color='k', linestyle='--', label='Uncontrolled u=0')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Control Input u')
    axs[1].set_title('MPC Control Input with Uncertainty Bounds')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()