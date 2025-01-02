# simulate_2dof.py

import numpy as np
from scipy.integrate import solve_ivp

def simulate_true(m1, m2, c1, c2, k1, k2, theta_0_1, theta_0_2,
                  x0, t, U, noise_array_1, noise_array_2):
    """
    Simulate the true 2DOF system with known parameters plus noise.
    
    States: X = [x1, v1, x2, v2]
    Derivatives:
      x1_dot = v1
      v1_dot = (theta_0_1 - c1*v1 - k1*x1 - c2*(v1 - v2) - k2*(x1 - x2))/m1 + noise_1
      x2_dot = v2
      v2_dot = (theta_0_2 + u - c2*(v2 - v1) - k2*(x2 - x1))/m2 + noise_2

    Returns
    -------
    X_true : np.ndarray, shape (len(t), 4)
        The simulated trajectory over time.
    """
    X_true = np.zeros((len(t), 4))
    X_true[0] = x0

    def dynamics(t_val, state, u_val, n1, n2):
        x1, v1, x2, v2 = state
        dx1 = v1
        dx2 = v2
        dv1 = (theta_0_1 - c1*v1 - k1*x1
               - c2*(v1 - v2) - k2*(x1 - x2))/m1 + n1
        dv2 = (theta_0_2 + u_val
               - c2*(v2 - v1) - k2*(x2 - x1))/m2 + n2
        return [dx1, dv1, dx2, dv2]

    for i in range(len(t)-1):
        dt_span = [t[i], t[i+1]]
        t_eval = [t[i+1]]
        sol = solve_ivp(
            dynamics,
            dt_span,
            X_true[i],
            args=(U[i], noise_array_1[i], noise_array_2[i]),
            method='RK45',
            t_eval=t_eval
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed at step {i}: {sol.message}")
        X_true[i+1] = sol.y[:, -1]

    return X_true



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
        + [sigma_epsilon_1, sigma_epsilon_2]
    )
    return true_params