# simulate_2dof_linear_ou.py

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



def generate_ornstein_uhlenbeck_noise(t, theta=1.0, sigma=0.1, mu=0.0, x0=0.0, seed=None):
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
        dt = t[i+1] - t[i]
        dW = np.random.normal(0, np.sqrt(dt))
        ou[i+1] = ou[i] + theta*(mu - ou[i]) * dt + sigma * dW
    return ou


def simulate_true(
    m1, m2, c1, c2, k1, k2,
    theta_0_1, theta_0_2,
    x0, t, U,
    noise_type=None,
    sigma_epsilon_1=0.1,
    sigma_epsilon_2=0.1,
    # Optional parameters for OU/Brownian
    theta_ou=1.0,        # OU mean-reversion rate
    mu_ou=0.0,           # OU mean
    x0_ou1=0.0,          # OU initial for mass1
    x0_ou2=0.0,          # OU initial for mass2
    seed=None
):
    """
    Simulate the 2DOF linear system with a chosen noise type:
      - noise_type="brownian" => Brownian motion for each mass
      - noise_type="ou"       => Ornstein-Uhlenbeck for each mass
      - noise_type=None       => zero noise
      or adapt as needed.

    States: X = [x1, v1, x2, v2].
    Derivatives:
      x1_dot = v1
      v1_dot = (theta_0_1 - c1*v1 - k1*x1 - c2*(v1 - v2) - k2*(x1 - x2))/m1 + noise_1(t)
      x2_dot = v2
      v2_dot = (theta_0_2 + u - c2*(v2 - v1) - k2*(x2 - x1))/m2 + noise_2(t)
    """

    # ------------------------------------------------------------------
    # A) Generate noise arrays if requested
    # ------------------------------------------------------------------

    if noise_type == "ou":
        noise_array_1 = generate_ornstein_uhlenbeck_noise(
            t, theta=theta_ou, sigma=sigma_epsilon_1, mu=mu_ou, x0=x0_ou1, seed=seed
        )
        noise_array_2 = generate_ornstein_uhlenbeck_noise(
            t, theta=theta_ou, sigma=sigma_epsilon_2, mu=mu_ou, x0=x0_ou2, seed=seed+1 if seed else None
        )
        
        # Plot the discrete and continuous noise
        plt.figure(figsize=(10, 6))
        plt.plot(t, noise_array_1, '-', label="Discrete Noise (Original Points)")
        plt.xlabel("Time")
        plt.ylabel("Noise Value")
        plt.title("Interpolated Continuous Noise")
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.plot(t, noise_array_2, '-', label="Discrete Noise (Original Points)")
        plt.xlabel("Time")
        plt.ylabel("Noise Value")
        plt.title("Interpolated Continuous Noise")
        plt.legend()
        plt.show()
    else:
        # Default = zero noise
        noise_array_1 = np.zeros_like(t)
        noise_array_2 = np.zeros_like(t)

    
    # ------------------------------------------------------------------
    # B) ODE definition with time interpolation of noise + control
    # ------------------------------------------------------------------
    def dynamics(t_val, state):
        x1, v1, x2, v2 = state

        # Interpolate control at time t_val
        u_val = np.interp(t_val, t, U)

        # Interpolate noise at time t_val
        n1 = np.interp(t_val, t, noise_array_1)
        n2 = np.interp(t_val, t, noise_array_2)

        dx1 = v1
        dx2 = v2
        dv1 = (
            theta_0_1
            - c1*v1
            - k1*x1
            - c2*(v1 - v2)
            - k2*(x1 - x2)
        )/m1 + n1
        dv2 = (
            theta_0_2
            + u_val
            - c2*(v2 - v1)
            - k2*(x2 - x1)
        )/m2 + n2
        return [dx1, dv1, dx2, dv2]

    # ------------------------------------------------------------------
    # C) Solve over [t[0], t[-1]] with solve_ivp
    # ------------------------------------------------------------------
    sol = solve_ivp(
        fun=dynamics,
        t_span=(t[0], t[-1]),
        y0=x0,
        t_eval=t,         # we want solution at each step in t
        method='RK45'
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    X_true = sol.y.T  # shape (len(t), 4)

    # D) Optionally compute the derivatives array
    X_dot_true = np.zeros_like(X_true)
    for i, t_val in enumerate(t):
        dx = dynamics(t_val, X_true[i])
        X_dot_true[i] = dx

    return X_true, X_dot_true





def compute_true_coeffs(
    m1, m2, c1, c2, k1, k2,
    theta_0_1, theta_0_2,
    sigma_epsilon_1, sigma_epsilon_2,
    feature_names
):
    """
    For a known 2DOF system, compute the "true" parameter vector 
      [theta_0_1, coeffs_for_v1_dot..., theta_0_2, coeffs_for_v2_dot..., sigma_esp1, sigma_esp2].
    The indexing of each feature in v1_dot, v2_dot is determined by 'feature_names'.

    We assume a linear physical model:
        v1_dot = theta_0_1
                 - (c1 + c2)/m1 * v1
                 + c2/m1 * v2
                 - (k1 + k2)/m1 * x1
                 + k2/m1 * x2

        v2_dot = theta_0_2
                 + c2/m2 * v1
                 - c2/m2 * v2
                 + k2/m2 * x1
                 - k2/m2 * x2
                 + 1/m2  * u

    All other terms (like x1^2, x1*v1, etc.) => 0.

    Parameters
    ----------
    feature_names : list of str
        Must match exactly the library used to create Theta.

    Returns
    -------
    true_params : list of floats
        [theta_0_1, <coeffs for v1_dot>, theta_0_2, <coeffs for v2_dot>, 
         sigma_epsilon_1, sigma_epsilon_2].
    """
    # Initialize all features to 0 for each equation
    coeffs_v1 = np.zeros(len(feature_names)-1)
    coeffs_v2 = np.zeros(len(feature_names)-1)


    coeffs_v1[0] = -(k1 + k2)/m1
    coeffs_v1[1] = -(c1 + c2)/m1
    coeffs_v1[2] = k2/m1
    coeffs_v1[3] = c2/m1
    # i_u => 0 for v1_dot
    
    # v2_dot
    coeffs_v2[0] = k2/m2
    coeffs_v2[1] = c2/m2
    coeffs_v2[2] = -(k2/m2)
    coeffs_v2[3] = -(c2/m2)
    coeffs_v2[4]  = 1.0/m2

    # Combine into the final parameter vector
    true_params = (
        [theta_0_1]
        + list(coeffs_v1)
        + [theta_0_2]
        + list(coeffs_v2)
    )
    return true_params