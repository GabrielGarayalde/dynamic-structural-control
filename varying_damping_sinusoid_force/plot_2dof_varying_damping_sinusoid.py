import matplotlib.pyplot as plt
from sindy_2dof_varying_stiffness_sinusoid import predict_sindy_trajectory

def plot_mpc_results_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot the Uncontrolled Trajectory and MPC Mean Trajectory for a 2DOF system
    with *single* control input alpha(t).
    
    Parameters
    ----------
    t : (N,)
    X_uncontrolled : (N,4)
    X_mean : (N,4)
    X_std : (N,4)
    U_mean : (N-1,)
    U_std : (N-1,)
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
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("States (Mass 1)")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("State Comparison (Mass 1): Uncontrolled vs MPC")

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
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("States (Mass 2)")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title("State Comparison (Mass 2): Uncontrolled vs MPC")

    # 3) Control Input alpha(t)
    axs[2].plot(t[:-1], U_mean, 'b-', label="MPC Mean alpha(t)")
    axs[2].fill_between(
        t[:-1],
        U_mean - 2*U_std,
        U_mean + 2*U_std,
        color='b', alpha=0.2, label="95% CI alpha"
    )
    axs[2].axhline(0, color='k', linestyle='--')
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Control Input alpha(t)")
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_title("MPC Control Input (alpha) with Uncertainty Bounds")

    plt.tight_layout()
    plt.show()


def plot_mpc_results_phase_single_control(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot phase-space (x vs. v) for both masses: uncontrolled vs. MPC (mean),
    plus the time history of alpha(t) with uncertainty bounds.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.ravel()

    # 1) Phase space for Mass 1 (x1,v1)
    axs[0].plot(X_uncontrolled[:, 0], X_uncontrolled[:, 1], 'g-', label='Uncontrolled')
    axs[0].plot(X_mean[:, 0], X_mean[:, 1], 'r-', label='MPC Mean')
    # The fill_between trick is less straightforward in phase-space if x1 is the variable.
    # Instead, we could do a param-based approach, but for simplicity let's skip.
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
    axs[2].set_ylabel("Control Input alpha")
    axs[2].set_title("alpha(t) vs. Time")
    axs[2].legend()
    axs[2].grid(True)

    # (Optional) you could add a subplot for something else, or just leave blank.
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()



def plot_true_vs_estimated_model(t, X_true, x0, alpha_array, sin_forcing_array, fitted_model, title_prefix=""):
    """
    Compare the true vs. discovered SINDy model trajectories.
    We pass alpha(t) and sin_forcing(t) to the model (2 inputs).
    """
    X_est = predict_sindy_trajectory(
        fitted_model, x0, t, alpha_array, sin_forcing_array
    )

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    labels = ['x1','v1','x2','v2']
    for i in range(4):
        axs.flat[i].plot(t, X_true[:, i], 'k-', label='True')
        axs.flat[i].plot(t, X_est[:, i], 'r--', label='Estimated')
        axs.flat[i].set_title(f"{title_prefix} - {labels[i]}(t)")
        axs.flat[i].grid(True)
        if i == 0:
            axs.flat[i].legend()
    plt.tight_layout()
    plt.show()
