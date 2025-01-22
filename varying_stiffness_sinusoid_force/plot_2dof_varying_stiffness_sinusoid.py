import matplotlib.pyplot as plt
import numpy as np
from sindy_2dof_varying_damping_sinusoid import predict_sindy_trajectory

def plot_true_vs_estimated_model(t, X_true, x0,
                                 alpha_array, sin_forcing_array,
                                 fitted_model,
                                 title_prefix=""):
    """
    Plot the true system trajectory vs. discovered model
    under alpha(t) + sinusoidal forcing(t).
    """
    X_est = predict_sindy_trajectory(fitted_model, x0, t,
                                     alpha_array, sin_forcing_array)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    labels = ["x1","v1","x2","v2"]
    for i in range(4):
        axs.flat[i].plot(t, X_true[:, i], 'k-', label='True')
        axs.flat[i].plot(t, X_est[:, i], 'r--', label='Estimated')
        axs.flat[i].grid(True)
        axs.flat[i].set_title(f"{title_prefix}: {labels[i]}")
        if i == 0:
            axs.flat[i].legend()

    plt.tight_layout()
    plt.show()


def plot_mpc_results_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot the Uncontrolled Trajectory vs. the Bayesian MPC result
    for a 2DOF system with alpha(t) as single control + known sinusoidal forcing.
    """
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))

    # 1) Mass1 => x1, v1
    axs[0].plot(t, X_uncontrolled[:,0], 'g-', label='Uncontrolled x1')
    axs[0].plot(t, X_uncontrolled[:,1], 'g--', label='Uncontrolled v1')
    axs[0].plot(t, X_mean[:,0], 'r-', label='MPC x1 (mean)')
    axs[0].plot(t, X_mean[:,1], 'r--', label='MPC v1 (mean)')
    axs[0].fill_between(
        t,
        X_mean[:,0] - 2*X_std[:,0],
        X_mean[:,0] + 2*X_std[:,0],
        alpha=0.2, color='r'
    )
    axs[0].fill_between(
        t,
        X_mean[:,1] - 2*X_std[:,1],
        X_mean[:,1] + 2*X_std[:,1],
        alpha=0.2, color='r'
    )
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_ylabel("States (Mass1)")

    # 2) Mass2 => x2, v2
    axs[1].plot(t, X_uncontrolled[:,2], 'g-', label='Uncontrolled x2')
    axs[1].plot(t, X_uncontrolled[:,3], 'g--', label='Uncontrolled v2')
    axs[1].plot(t, X_mean[:,2], 'r-', label='MPC x2 (mean)')
    axs[1].plot(t, X_mean[:,3], 'r--', label='MPC v2 (mean)')
    axs[1].fill_between(
        t,
        X_mean[:,2] - 2*X_std[:,2],
        X_mean[:,2] + 2*X_std[:,2],
        alpha=0.2, color='r'
    )
    axs[1].fill_between(
        t,
        X_mean[:,3] - 2*X_std[:,3],
        X_mean[:,3] + 2*X_std[:,3],
        alpha=0.2, color='r'
    )
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_ylabel("States (Mass2)")

    # 3) alpha(t)
    axs[2].plot(t[:-1], U_mean, 'b-', label='MPC Mean alpha(t)')
    axs[2].fill_between(
        t[:-1],
        U_mean - 2*U_std,
        U_mean + 2*U_std,
        alpha=0.2, color='b'
    )
    axs[2].axhline(0, color='k', linestyle='--')
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("alpha(t)")

    plt.tight_layout()
    plt.show()


def plot_mpc_results_phase_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot phase space for mass1, mass2, and alpha(t) over time.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.ravel()

    # Phase space: mass1
    axs[0].plot(X_uncontrolled[:,0], X_uncontrolled[:,1], 'g-', label='Uncontrolled')
    axs[0].plot(X_mean[:,0], X_mean[:,1], 'r-', label='MPC Mean')
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("v1")
    axs[0].set_title("Phase (Mass1)")
    axs[0].grid(True)
    axs[0].legend()

    # Phase space: mass2
    axs[1].plot(X_uncontrolled[:,2], X_uncontrolled[:,3], 'g-', label='Uncontrolled')
    axs[1].plot(X_mean[:,2], X_mean[:,3], 'r-', label='MPC Mean')
    axs[1].set_xlabel("x2")
    axs[1].set_ylabel("v2")
    axs[1].set_title("Phase (Mass2)")
    axs[1].grid(True)
    axs[1].legend()

    # alpha(t) over time
    axs[2].plot(t[:-1], U_mean, 'b-', label="alpha(t) Mean")
    axs[2].fill_between(
        t[:-1],
        U_mean - 2*U_std,
        U_mean + 2*U_std,
        color='b', alpha=0.2
    )
    axs[2].axhline(0, color='k', linestyle='--')
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("alpha(t)")
    axs[2].set_title("Control Input vs. Time")
    axs[2].legend()
    axs[2].grid(True)

    axs[3].axis('off')
    plt.tight_layout()
    plt.show()
