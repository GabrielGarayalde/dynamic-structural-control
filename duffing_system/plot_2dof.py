import matplotlib.pyplot as plt
from sindy_2dof import predict_sindy_trajectory

def plot_mpc_results(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot the Uncontrolled Trajectory and MPC Mean Trajectory for a 2DOF system
    with *two* distinct control inputs U_mean[:,0], U_mean[:,1].
    
    Parameters
    ----------
    t : 1D array of time, shape (nt,)
    X_uncontrolled : shape (nt, 4)
        The uncontrolled trajectory [x1, v1, x2, v2].
    X_mean : shape (nt, 4)
        The mean MPC trajectory.
    X_std : shape (nt, 4)
        The std dev of the MPC trajectory across multiple runs.
    U_mean : shape (nt-1, 2)
        The mean MPC control input at each time step for both masses
        (u1 in column 0, u2 in column 1).
    U_std : shape (nt-1, 2)
        The std dev of the MPC control input for both masses.
    """
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))

    # -------------------------
    # 1) Mass 1 states: x1, v1
    # -------------------------
    axs[0].plot(t, X_uncontrolled[:, 0], 'g-', label='Uncontrolled x1')
    axs[0].plot(t, X_uncontrolled[:, 1], 'g--', label='Uncontrolled v1')
    axs[0].plot(t, X_mean[:, 0], 'r-', label='MPC Mean x1')
    axs[0].plot(t, X_mean[:, 1], 'r--', label='MPC Mean v1')
    axs[0].fill_between(
        t,
        X_mean[:, 0] - 2*X_std[:, 0],
        X_mean[:, 0] + 2*X_std[:, 0],
        color='r', alpha=0.2, label='95% CI x1'
    )
    axs[0].fill_between(
        t,
        X_mean[:, 1] - 2*X_std[:, 1],
        X_mean[:, 1] + 2*X_std[:, 1],
        color='r', alpha=0.2, label='95% CI v1'
    )
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('States (Mass 1)')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title('State Comparison (Mass 1): Uncontrolled vs MPC')

    # -------------------------
    # 2) Mass 2 states: x2, v2
    # -------------------------
    axs[1].plot(t, X_uncontrolled[:, 2], 'g-', label='Uncontrolled x2')
    axs[1].plot(t, X_uncontrolled[:, 3], 'g--', label='Uncontrolled v2')
    axs[1].plot(t, X_mean[:, 2], 'r-', label='MPC Mean x2')
    axs[1].plot(t, X_mean[:, 3], 'r--', label='MPC Mean v2')
    axs[1].fill_between(
        t,
        X_mean[:, 2] - 2*X_std[:, 2],
        X_mean[:, 2] + 2*X_std[:, 2],
        color='r', alpha=0.2, label='95% CI x2'
    )
    axs[1].fill_between(
        t,
        X_mean[:, 3] - 2*X_std[:, 3],
        X_mean[:, 3] + 2*X_std[:, 3],
        color='r', alpha=0.2, label='95% CI v2'
    )
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('States (Mass 2)')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title('State Comparison (Mass 2): Uncontrolled vs MPC')

    # -------------------------
    # 3) Control Inputs: u1(t), u2(t)
    # -------------------------
    # Plot u1(t)
    axs[2].plot(t[:-1], U_mean[:, 0], 'b-', label='MPC Mean u1(t)')
    axs[2].fill_between(
        t[:-1],
        U_mean[:, 0] - 2*U_std[:, 0],
        U_mean[:, 0] + 2*U_std[:, 0],
        color='b', alpha=0.2, label='95% CI u1'
    )

    # Plot u2(t)
    axs[2].plot(t[:-1], U_mean[:, 1], 'm-', label='MPC Mean u2(t)')
    axs[2].fill_between(
        t[:-1],
        U_mean[:, 1] - 2*U_std[:, 1],
        U_mean[:, 1] + 2*U_std[:, 1],
        color='m', alpha=0.2, label='95% CI u2'
    )

    axs[2].axhline(0, color='k', linestyle='--', label='Uncontrolled (0,0)')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Control Inputs (u1, u2)')
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_title('MPC Control Inputs with Uncertainty Bounds')

    plt.tight_layout()
    plt.show()

def plot_mpc_results_phase(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot phase-space (x vs. v) for both masses: uncontrolled vs. MPC (mean),
    plus the time history of both control inputs with uncertainty bounds.
    
    Parameters
    ----------
    t : shape (nt,)
    X_uncontrolled : shape (nt, 4)
    X_mean : shape (nt, 4)
    X_std : shape (nt, 4)
    U_mean : shape (nt-1, 2)
    U_std : shape (nt-1, 2)
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.ravel()  # so we can index them as axs[0], axs[1], axs[2], axs[3]

    # ---------------------------------
    # 1) Phase space for Mass 1 (x1,v1)
    # ---------------------------------
    axs[0].plot(X_uncontrolled[:, 0], X_uncontrolled[:, 1], 'g-', label='Uncontrolled')
    axs[0].plot(X_mean[:, 0], X_mean[:, 1], 'r-', label='MPC Mean')
    axs[0].fill_between(
        X_mean[:, 0],
        X_mean[:, 1] - 2*X_std[:, 1],
        X_mean[:, 1] + 2*X_std[:, 1],
        color='r', alpha=0.2
    )
    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('v1')
    axs[0].set_title('Phase Space (Mass 1)')
    axs[0].legend()
    axs[0].grid(True)

    # ---------------------------------
    # 2) Phase space for Mass 2 (x2,v2)
    # ---------------------------------
    axs[1].plot(X_uncontrolled[:, 2], X_uncontrolled[:, 3], 'g-', label='Uncontrolled')
    axs[1].plot(X_mean[:, 2], X_mean[:, 3], 'r-', label='MPC Mean')
    axs[1].fill_between(
        X_mean[:, 2],
        X_mean[:, 3] - 2*X_std[:, 3],
        X_mean[:, 3] + 2*X_std[:, 3],
        color='r', alpha=0.2
    )
    axs[1].set_xlabel('x2')
    axs[1].set_ylabel('v2')
    axs[1].set_title('Phase Space (Mass 2)')
    axs[1].legend()
    axs[1].grid(True)

    # ---------------------------------
    # 3) Control Input #1 over time
    # ---------------------------------
    axs[2].plot(t[:-1], U_mean[:, 0], 'b-', label='u1(t) Mean')
    axs[2].fill_between(
        t[:-1],
        U_mean[:, 0] - 2*U_std[:, 0],
        U_mean[:, 0] + 2*U_std[:, 0],
        color='b', alpha=0.2, label='95% CI'
    )
    axs[2].axhline(0, color='k', linestyle='--')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Control Input u1')
    axs[2].set_title('u1(t) vs Time')
    axs[2].legend()
    axs[2].grid(True)

    # ---------------------------------
    # 4) Control Input #2 over time
    # ---------------------------------
    axs[3].plot(t[:-1], U_mean[:, 1], 'm-', label='u2(t) Mean')
    axs[3].fill_between(
        t[:-1],
        U_mean[:, 1] - 2*U_std[:, 1],
        U_mean[:, 1] + 2*U_std[:, 1],
        color='m', alpha=0.2, label='95% CI'
    )
    axs[3].axhline(0, color='k', linestyle='--')
    axs[3].set_xlabel('Time [s]')
    axs[3].set_ylabel('Control Input u2')
    axs[3].set_title('u2(t) vs Time')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()


def plot_true_vs_estimated_model(t, X_true, x0, U, fitted_model, title_prefix=""):
    """
    Plot the true system trajectory vs. the discovered SINDy model 
    for the same initial condition.
    
    Parameters
    ----------
    t : ndarray
        Time array
    X_true : ndarray, shape (len(t), 4)
        True states
    x0 : ndarray, shape (4,)
        Initial condition
    U : ndarray, shape (len(t),)
        Control input
    fitted_model : pysindy.SINDy
        Discovered model to be integrated via Euler
    title_prefix : str
        A label prefix for the plot title (e.g. "In-sample" or "Robustness check").
    """
    X_est = predict_sindy_trajectory(fitted_model, x0, t, U)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    state_labels = ['x1', 'v1', 'x2', 'v2']

    for i_state in range(4):
        axs[i_state].plot(t, X_true[:, i_state], 'k-', label='True', linewidth=2)
        axs[i_state].plot(t, X_est[:, i_state], 'r--', label='Discovered Model')
        axs[i_state].set_title(f"{title_prefix} - State: {state_labels[i_state]}")
        axs[i_state].set_xlabel("Time [s]")
        axs[i_state].set_ylabel("Value")
        axs[i_state].grid(True)

        if i_state == 0:
            axs[i_state].legend()

    plt.tight_layout()
    plt.show()