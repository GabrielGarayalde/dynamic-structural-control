import matplotlib.pyplot as plt


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