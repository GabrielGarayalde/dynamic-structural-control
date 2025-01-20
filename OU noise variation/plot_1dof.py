import matplotlib.pyplot as plt
import numpy as np
from sindy_1dof import predict_sindy_trajectory

###############################################################################
# 1) plot_mpc_results_1dof
###############################################################################
def plot_mpc_results_1dof(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot the Uncontrolled Trajectory and MPC Mean Trajectory for a 1DOF system
    with *one* control input u(t).
    
    Parameters
    ----------
    t : 1D array of time, shape (nt,)
    X_uncontrolled : shape (nt, 2)
        The uncontrolled trajectory [x, v].
    X_mean : shape (nt, 2)
        The mean MPC trajectory.
    X_std : shape (nt, 2)
        The std dev of the MPC trajectory across multiple runs.
    U_mean : shape (nt-1,)
        The mean MPC control input at each time step.
    U_std : shape (nt-1,)
        The std dev of the MPC control input at each time step.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # -------------------------
    # 1) States: x, v
    # -------------------------
    axs[0].plot(t, X_uncontrolled[:, 0], 'g-',  label='Uncontrolled x')
    axs[0].plot(t, X_uncontrolled[:, 1], 'g--', label='Uncontrolled v')

    axs[0].plot(t, X_mean[:, 0], 'r-',  label='MPC Mean x')
    axs[0].plot(t, X_mean[:, 1], 'r--', label='MPC Mean v')

    # 95% confidence intervals for x
    axs[0].fill_between(
        t,
        X_mean[:, 0] - 2 * X_std[:, 0],
        X_mean[:, 0] + 2 * X_std[:, 0],
        color='r', alpha=0.2, label='95% CI (x)'
    )
    # 95% confidence intervals for v
    axs[0].fill_between(
        t,
        X_mean[:, 1] - 2 * X_std[:, 1],
        X_mean[:, 1] + 2 * X_std[:, 1],
        color='r', alpha=0.2, label='95% CI (v)'
    )

    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('States: x, v')
    axs[0].grid(True)
    axs[0].set_title('1DOF State Comparison: Uncontrolled vs MPC')
    axs[0].legend()

    # -------------------------
    # 2) Control Input: u(t)
    # -------------------------
    # Plot only up to t[:-1] because the last time step does not have a control
    axs[1].plot(t[:-1], U_mean, 'b-', label='MPC Mean u(t)')
    axs[1].fill_between(
        t[:-1],
        U_mean - 2 * U_std,
        U_mean + 2 * U_std,
        color='b', alpha=0.2, label='95% CI (u)'
    )

    axs[1].axhline(0, color='k', linestyle='--', label='Uncontrolled u=0')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Control Input u(t)')
    axs[1].grid(True)
    axs[1].set_title('MPC Control Input with Uncertainty Bounds')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


###############################################################################
# 2) plot_mpc_results_phase_1dof
###############################################################################
def plot_mpc_results_phase_1dof(t, X_uncontrolled, X_mean, X_std, U_mean, U_std):
    """
    Plot phase-space (x vs. v) for a 1DOF system: uncontrolled vs. MPC (mean),
    plus the time history of the control input with uncertainty bounds.
    
    Parameters
    ----------
    t : shape (nt,)
    X_uncontrolled : shape (nt, 2)
    X_mean : shape (nt, 2)
    X_std : shape (nt, 2)
    U_mean : shape (nt-1,)
    U_std : shape (nt-1,)
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs = axs.ravel()

    # ---------------------------------
    # 1) Phase space (x vs. v)
    # ---------------------------------
    axs[0].plot(X_uncontrolled[:, 0], X_uncontrolled[:, 1],
                'g-', label='Uncontrolled')
    axs[0].plot(X_mean[:, 0], X_mean[:, 1],
                'r-', label='MPC Mean')
    # Shade ±2 std in the v-direction, if you want
    # (There is no direct "fill_between" for x-y plotting, so
    #  an alternative is to plot error bars at discrete intervals.)
    # For simplicity, we'll skip the fill for phase space,
    # or you could do a parametric approach.

    axs[0].set_xlabel('x')
    axs[0].set_ylabel('v')
    axs[0].set_title('Phase Space (1DOF)')
    axs[0].legend()
    axs[0].grid(True)

    # ---------------------------------
    # 2) Control Input u(t) over time
    # ---------------------------------
    axs[1].plot(t[:-1], U_mean, 'b-', label='u(t) Mean')
    axs[1].fill_between(
        t[:-1],
        U_mean - 2 * U_std,
        U_mean + 2 * U_std,
        color='b', alpha=0.2, label='95% CI'
    )
    axs[1].axhline(0, color='k', linestyle='--')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Control Input u(t)')
    axs[1].set_title('Control Input vs Time')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


###############################################################################
# 3) plot_true_vs_estimated_model_1dof
###############################################################################
def plot_true_vs_estimated_model(t, X_true, x0, u, fitted_model,                                       title_prefix=""):
    """
    Plot the true system trajectory vs. the discovered SINDy model 
    for the same initial condition in a 1DOF setup.

    Parameters
    ----------
    t : ndarray of shape (nt,)
        Time array
    X_true : ndarray, shape (nt, 2)
        True states, [x(t), v(t)]
    x0 : ndarray, shape (2,)
        Initial condition
    u : ndarray, shape (nt,)
        Control input
    fitted_model : pysindy.SINDy
        Discovered model to be integrated (predict_function uses it)
    predict_function : callable
        A function that predicts the 1DOF trajectory, e.g.:
          predict_sindy_trajectory_1dof(fitted_model, x0, t, u)
    title_prefix : str
        Label prefix for the plot title
    """
    # 1) Get SINDy-predicted trajectory
    X_est = predict_sindy_trajectory(fitted_model, x0, t, u)  # shape (nt, 2)

    # 2) Plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # -- x(t)
    axs[0].plot(t, X_true[:, 0], 'k-', label='True x(t)', linewidth=2)
    axs[0].plot(t, X_est[:, 0], 'r--', label='SINDy x(t)')
    axs[0].set_title(f"{title_prefix} - Displacement: x(t)")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("x(t)")
    axs[0].grid(True)
    axs[0].legend()

    # -- v(t)
    axs[1].plot(t, X_true[:, 1], 'k-', label='True v(t)', linewidth=2)
    axs[1].plot(t, X_est[:, 1], 'r--', label='SINDy v(t)')
    axs[1].set_title(f"{title_prefix} - Velocity: v(t)")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("v(t)")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()
