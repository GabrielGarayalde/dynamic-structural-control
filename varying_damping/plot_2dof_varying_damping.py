import matplotlib.pyplot as plt
import numpy as np

from sindy_2dof_varying_damping import predict_sindy_trajectory

def plot_true_vs_estimated_model(t, X_true, x0, U, fitted_model, title_prefix=""):
    """
    Plot the true system trajectory vs. discovered SINDy model.
    U is alpha(t) for damping distribution.
    """
    X_est = predict_sindy_trajectory(fitted_model, x0, t, U)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    state_labels = ['x1', 'v1', 'x2', 'v2']

    for i_state in range(4):
        axs[i_state].plot(t, X_true[:, i_state], 'k-', label='True')
        axs[i_state].plot(t, X_est[:, i_state], 'r--', label='Estimated')
        axs[i_state].set_title(f"{title_prefix} - State: {state_labels[i_state]}")
        axs[i_state].set_xlabel("Time [s]")
        axs[i_state].grid(True)
        if i_state == 0:
            axs[i_state].legend()

    plt.tight_layout()
    plt.show()


def plot_mpc_results_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot the Uncontrolled Trajectory vs. MPC Mean Trajectory
    for a 2DOF system with single damping-ratio control alpha(t).
    """
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))

    # 1) Mass 1 states: x1, v1
    axs[0].plot(t, X_uncontrolled[:, 0], 'g-', label='Uncontrolled x1')
    axs[0].plot(t, X_uncontrolled[:, 1], 'g--', label='Uncontrolled v1')
    axs[0].plot(t, X_mean[:, 0], 'r-', label='MPC Mean x1')
    axs[0].plot(t, X_mean[:, 1], 'r--', label='MPC Mean v1')
    axs[0].fill_between(
        t,
        X_mean[:, 0] - 2*X_std[:, 0],
        X_mean[:, 0] + 2*X_std[:, 0],
        color='r', alpha=0.2
    )
    axs[0].fill_between(
        t,
        X_mean[:, 1] - 2*X_std[:, 1],
        X_mean[:, 1] + 2*X_std[:, 1],
        color='r', alpha=0.2
    )
    axs[0].set_ylabel("States (Mass 1)")
    axs[0].legend()
    axs[0].grid(True)

    # 2) Mass 2 states: x2, v2
    axs[1].plot(t, X_uncontrolled[:, 2], 'g-', label='Uncontrolled x2')
    axs[1].plot(t, X_uncontrolled[:, 3], 'g--', label='Uncontrolled v2')
    axs[1].plot(t, X_mean[:, 2], 'r-', label='MPC Mean x2')
    axs[1].plot(t, X_mean[:, 3], 'r--', label='MPC Mean v2')
    axs[1].fill_between(
        t,
        X_mean[:, 2] - 2*X_std[:, 2],
        X_mean[:, 2] + 2*X_std[:, 2],
        color='r', alpha=0.2
    )
    axs[1].fill_between(
        t,
        X_mean[:, 3] - 2*X_std[:, 3],
        X_mean[:, 3] + 2*X_std[:, 3],
        color='r', alpha=0.2
    )
    axs[1].set_ylabel("States (Mass 2)")
    axs[1].legend()
    axs[1].grid(True)

    # 3) Control Input alpha(t)
    axs[2].plot(t[:-1], U_mean, 'b-', label="MPC Mean alpha(t)")
    axs[2].fill_between(
        t[:-1],
        U_mean - 2*U_std,
        U_mean + 2*U_std,
        color='b', alpha=0.2,
        label="95% CI alpha"
    )
    axs[2].axhline(0, color='k', linestyle='--')
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Damping Ratio alpha(t)")
    axs[2].legend()
    axs[2].grid(True)
    plt.tight_layout()
    plt.show()


def plot_mpc_results_phase_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot phase-space (x vs. v) for both masses: uncontrolled vs. MPC (mean).
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.ravel()

    # 1) Phase space for Mass 1 (x1,v1)
    axs[0].plot(X_uncontrolled[:, 0], X_uncontrolled[:, 1], 'g-', label='Uncontrolled')
    axs[0].plot(X_mean[:, 0], X_mean[:, 1], 'r-', label='MPC Mean')
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("v1")
    axs[0].set_title("Phase Space (Mass 1)")
    axs[0].legend()
    axs[0].grid(True)

    # 2) Phase space for Mass 2 (x2,v2)
    axs[1].plot(X_uncontrolled[:, 2], X_uncontrolled[:, 3], 'g-', label='Uncontrolled')
    axs[1].plot(X_mean[:, 2], X_mean[:, 3], 'r-', label='MPC Mean')
    axs[1].set_xlabel("x2")
    axs[1].set_ylabel("v2")
    axs[1].set_title("Phase Space (Mass 2)")
    axs[1].legend()
    axs[1].grid(True)

    # 3) alpha(t) vs time
    axs[2].plot(t[:-1], U_mean, 'b-', label="alpha(t) Mean")
    axs[2].fill_between(
        t[:-1],
        U_mean - 2*U_std,
        U_mean + 2*U_std,
        color='b', alpha=0.2,
        label="95% CI"
    )
    axs[2].axhline(0, color='k', linestyle='--')
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Control alpha(t)")
    axs[2].set_title("Alpha(t) vs. Time")
    axs[2].legend()
    axs[2].grid(True)

    axs[3].axis('off')
    plt.tight_layout()
    plt.show()
