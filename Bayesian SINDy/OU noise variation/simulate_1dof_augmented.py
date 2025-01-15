import numpy as np

def simulate_true_with_hidden_z(
    m, c, k,
    theta_0,         # external offset
    x0, v0, z0,      # initial conditions for x, v, z
    t, u,
    theta_z=1.0,     # OU mean-reversion rate
    mu_z=0.0,        # OU long-term mean
    sigma_z=0.1,     # OU diffusion
    seed=None
):
    """
    Simulate a 1DOF system with a *hidden* Ornstein-Uhlenbeck state z.

    True States: [x, v, z], with:
      x_dot = v
      v_dot = (theta_0 + u(t) - c*v - k*x)/m + z
      z_dot = theta_z*(mu_z - z) + sigma_z * dW/dt

    BUT we only return (x, v) and (x_dot, v_dot) to emulate that z is unobservable.

    Integration approach: Euler-Maruyama in a step-by-step loop.

    Parameters
    ----------
    m,c,k : float
        System mass, damping, stiffness
    theta_0 : float
        Constant offset in v_dot
    x0,v0,z0 : float
        Initial conditions for x, v, z
    t : ndarray of shape (N,)
        Time array
    u : ndarray of shape (N,)  or (N,1)
        Single control input over time
    theta_z, mu_z, sigma_z : float
        OU parameters for the hidden z
    seed : int or None
        Seed for reproducible noise (optional)

    Returns
    -------
    X_obs : ndarray, shape (N, 2)
        The observed states [x, v] at each time step
    X_dot_obs : ndarray, shape (N, 2)
        The derivatives [x_dot, v_dot], ignoring z-dot since we cannot measure it
    """
    if seed is not None:
        np.random.seed(seed)

    if u.ndim == 2:  # shape (N,1) => flatten
        u = u.squeeze()

    N = len(t)
    dt_array = np.diff(t)

    # Full hidden state includes z
    X_full = np.zeros((N, 3))  # columns => x, v, z
    X_full[0] = [x0, v0, z0]

    # For returning, we only keep [x, v] plus [x_dot, v_dot]
    X_obs = np.zeros((N, 2))      # measured states
    X_dot_obs = np.zeros((N, 2))  # measured derivatives

    X_obs[0] = [x0, v0]

    for i in range(N - 1):
        dt_local = dt_array[i]
        x_i, v_i, z_i = X_full[i]

        # OU increment
        dW = np.random.normal(0, np.sqrt(dt_local))

        # Euler-Maruyama updates
        # 1) x_{n+1}
        x_dot_i = v_i
        x_next = x_i + dt_local * x_dot_i

        # 2) v_{n+1}
        v_dot_i = (theta_0 + u[i] - c*v_i - k*x_i) / m + z_i
        v_next = v_i + dt_local * v_dot_i

        # 3) z_{n+1}
        z_drift = theta_z*(mu_z - z_i)
        z_next = z_i + dt_local * z_drift + sigma_z*dW

        # Save to X_full
        X_full[i+1] = [x_next, v_next, z_next]

        # Also store obs states + derivatives
        X_obs[i] = [x_i, v_i]
        X_dot_obs[i] = [x_dot_i, v_dot_i]

    # Last step
    X_obs[-1] = X_full[-1, :2]
    # Approx derivative => same as second-to-last
    X_dot_obs[-1] = X_dot_obs[-2]

    return X_obs, X_dot_obs
