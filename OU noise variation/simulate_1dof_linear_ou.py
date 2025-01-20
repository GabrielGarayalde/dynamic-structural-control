import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def generate_ornstein_uhlenbeck_noise_1dof(
    t, theta=1.0, sigma=0.1, mu=0.0, x0=0.0, seed=None
):
    """
    Generate an OU process X(t) that follows:
        dX = theta*(mu - X) dt + sigma dW
    Discretized with Euler-Maruyama. X(0) = x0.
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(t)
    ou = np.zeros(N)
    ou[0] = x0
    for i in range(N - 1):
        dt_local = t[i+1] - t[i]
        dW = np.random.normal(0, np.sqrt(dt_local))
        ou[i+1] = ou[i] + theta*(mu - ou[i]) * dt_local + sigma * dW
    return ou

def simulate_true(
    m, c, k,
    theta_0,
    X0, t, u,
    mu_ou=0.0,        # OU mean
    sigma_epsilon=0.1,
    theta_ou=1.0,     # OU mean-reversion rate
    seed=None
):
    """
    Simulate a 1DOF system with known parameters plus noise.

    State: X = [x, v].
    Equations:
        x_dot = v
        v_dot = [ (theta_0 + u(t)) - c*v - k*x ] / m + noise

    Noise is an OU process, one for the v_dot equation.
    """
    # Prepare arrays to store state at each time step
    X_sim = np.zeros((len(t), 2))
    X_sim[0] = X0

    # Generate OU noise
    noise_array = generate_ornstein_uhlenbeck_noise_1dof(
        t, theta=theta_ou, sigma=sigma_epsilon, mu=mu_ou, x0=mu_ou, seed=seed
    )

    # (Optional) quick plot of noise for demonstration
    plt.figure(figsize=(10, 4))
    plt.plot(t, noise_array, 'k-', label="Noise")
    plt.xlabel("Time")
    plt.ylabel("Noise Value")
    plt.title("1DOF Noise Over Time")
    plt.legend()
    plt.show()

    def dynamics(t_val, state, u_val, n):
        x, v = state
        dx = v
        dv = (theta_0 + u_val - c*v - k*x) / m + n
        return [dx, dv]

    for i in range(len(t) - 1):
        dt_span = [t[i], t[i+1]]
        t_eval = [t[i+1]]

        # control input
        u_i = u[i]
        # noise
        n_i = noise_array[i]

        sol = solve_ivp(
            dynamics,
            dt_span,
            X_sim[i],
            args=(u_i, n_i),
            method='RK45',
            t_eval=t_eval
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed at step {i}: {sol.message}")
        X_sim[i+1] = sol.y[:, -1]

    # Compute derivatives X_dot from the simulated trajectory
    X_dot_sim = np.zeros_like(X_sim)
    for i in range(len(t) - 1):
        x_i, v_i = X_sim[i]
        dx_i = v_i
        dv_i = (theta_0 + u[i] - c*v_i - k*x_i) / m + noise_array[i]
        X_dot_sim[i] = [dx_i, dv_i]

    # Copy the second-to-last derivative to the last step
    X_dot_sim[-1] = X_dot_sim[-2]

    return X_sim, X_dot_sim


def compute_true_coeffs(
    m, c, k,
    theta_0, sigma_epsilon,
    feature_names
):
    """
    For a known 1DOF system with input u, compute the "true" parameter vector for v_dot.

    v_dot = theta_0/m
            - c/m * v
            - k/m * x
            + 1/m * u
    (Plus noise, which is separate.)

    The library is typically:
       1 (bias),
       x, v, u,
       plus any polynomial cross-terms if poly_degree>1, etc.

    We'll set the correct linear coefficients and 0 for cross-terms.
    """
    # e.g. feature_names might be ['1', 'x0', 'x1', 'u0', 'x0^2', ...]
    # We'll assume:
    #   'x0' => x
    #   'x1' => v
    #   'u0' => u
    # adjust as needed based on your naming

    # Make an array of zeros, length = len(feature_names) - 1 (excluding '1'?)
    # but in PySINDy, '1' might be the first feature, then x, v, u, ...
    # We'll figure out the indices:
    #    '1'   => constant
    #    'x0'  => x
    #    'x1'  => v
    #    'u0'  => u
    # (plus possibly more cross-terms)
    n_features = len(feature_names)
    coeffs_v = np.zeros(n_features)

    # We find the indices:
    idx_bias = feature_names.index('1')       # bias
    idx_x    = feature_names.index('x0')      # may differ depending on naming
    idx_v    = feature_names.index('x1')
    idx_u    = feature_names.index('u0')      # if present

    # Set the known values:
    # v_dot = (theta_0 - c*v - k*x + u) / m
    coeffs_v[idx_bias] = theta_0 / m
    coeffs_v[idx_x]    = -k / m
    coeffs_v[idx_v]    = -c / m
    coeffs_v[idx_u]    = 1.0 / m  # if your library has 'u0'

    # Return just this single array (since 1DOF => only one row for v_dot)
    return coeffs_v
