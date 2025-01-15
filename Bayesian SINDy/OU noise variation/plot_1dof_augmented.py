import matplotlib.pyplot as plt
import numpy as np

def predict_sindy_trajectory_augmented(fitted_model, x0v0z0, t, u):
    """
    Predict the [x,v,z] trajectory from a SINDy model by simple Euler steps.

    Parameters
    ----------
    fitted_model : pysindy.SINDy
        Learned model with 3 states: x, v, z + 1 input: u
    x0v0z0 : array-like, shape (3,)
        [x(0), v(0), z(0)]
    t : array-like of shape (nt,)
    u : array-like of shape (nt,) or (nt,1)
        Single control input vs. time

    Returns
    -------
    X_pred : ndarray of shape (nt, 3)
        Predicted [x, v, z] at each time step
    """
    if u.ndim == 1:
        u = u.reshape(-1, 1)   # shape => (nt, 1) for PySINDy
    dt = t[1] - t[0]

    X_pred = np.zeros((len(t), 3))
    X_pred[0] = x0v0z0

    for i in range(len(t) - 1):
        state_now = X_pred[i].reshape(1, -1)  # shape (1,3)
        control_now = u[i].reshape(1, -1)     # shape (1,1)
        # predict x_dot, v_dot, z_dot from the SINDy model
        x_dot_pred = fitted_model.predict(state_now, u=control_now)[0]  # shape (3,)
        # Euler update
        X_pred[i+1] = X_pred[i] + dt * x_dot_pred

    return X_pred

def compare_sindy_vs_true_augmented(
    fitted_model,
    simulate_fn,
    m, c, k, theta_0,
    x0, v0, z0,
    t,
    u_original,
    # OU parameters
    theta_z=1.0,
    mu_z=0.0,
    sigma_z=0.1,
    seed=None,
    no_control=False,
    no_noise=False,
    title_prefix=""
):
    """
    Compare the true system vs. SINDy discovered model for a 1DOF + OU system.
    Allows toggling off control (u=0) and/or noise (sigma_z=0).

    Parameters
    ----------
    fitted_model : pysindy.SINDy
        The discovered SINDy model with states [x,v,z].
    simulate_fn : callable
        Your function to simulate the true system, e.g. simulate_true_augmented(...).
    predict_fn : callable
        The function to predict with SINDy, e.g. predict_sindy_trajectory_augmented(...).
    m,c,k,theta_0 : float
        Physical parameters
    x0,v0,z0 : float
        Initial conditions
    t : array-like
        Time array
    u_original : array-like
        The original control input to use (will override with 0 if no_control=True).
    theta_z, mu_z, sigma_z : float
        OU parameters
    seed : int or None
        Random seed for the true simulation if desired.
    no_control : bool
        If True, sets control input to 0 for the simulation and SINDy.
    no_noise : bool
        If True, sets sigma_z=0 in the true simulation.
        (SINDy prediction is always deterministic, so this just removes noise from the true model.)
    title_prefix : str
        A label to prepend to plot titles.
    """
    # ------------------------------
    # 1) Possibly zero-out control
    # ------------------------------
    if no_control:
        u = np.zeros_like(u_original)
    else:
        u = np.copy(u_original)

    # ------------------------------
    # 2) Possibly turn off OU noise in true sim
    # ------------------------------
    sigma_z_local = 0.0 if no_noise else sigma_z

    # ------------------------------
    # 3) Simulate TRUE system
    # ------------------------------
    X_true, _ = simulate_fn(
        m, c, k,
        theta_0,
        x0, v0, z0,
        t, u,
        theta_z=theta_z,
        mu_z=mu_z,
        sigma_z=sigma_z_local,
        seed=seed
    )
    # X_true shape => (len(t), 3) => columns [x,v,z]

    # ------------------------------
    # 4) Predict from SINDy
    # ------------------------------
    X_sindy = predict_sindy_trajectory_augmented(
        fitted_model,
        x0v0z0=[x0, v0, z0],
        t=t,
        u=u
    )

    # ------------------------------
    # 5) Plot Comparison
    # ------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    labels = ['x(t)', 'v(t)', 'z(t)']

    for i in range(3):
        axs[i].plot(t, X_true[:, i], 'k-', label='True System')
        axs[i].plot(t, X_sindy[:, i], 'r--', label='SINDy')
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        if i == 0:
            axs[i].legend()

    axs[-1].set_xlabel('Time [s]')
    plt.suptitle(f"{title_prefix} - Compare True vs. SINDy")
    plt.tight_layout()
    plt.show()

    return X_true, X_sindy
