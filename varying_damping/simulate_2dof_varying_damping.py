# simulate_2dof_varying_stiffness.py

import numpy as np
from scipy.integrate import solve_ivp



def simulate_true_varying_damping(
    m1, m2,
    c_total,
    k1, k2,
    theta_0_1, theta_0_2,
    x0, t,
    alpha_array,
    noise_array_1, noise_array_2
):
    """
    Simulate a 2DOF system where total damping c_total is split as:
        c1(t) = alpha(t)*c_total,
        c2(t) = (1 - alpha(t))*c_total.

    The stiffness k1, k2 remain constant.

    States: X = [x1, v1, x2, v2].

    Equations of motion:
      x1_dot = v1
      v1_dot = [theta_0_1
                - c1(t)*v1
                - k1*x1
                - c2(t)*(v1 - v2)
                - k2*(x1 - x2)] / m1 + noise_1

      x2_dot = v2
      v2_dot = [theta_0_2
                - c2(t)*(v2 - v1)
                - k2*(x2 - x1)] / m2 + noise_2

    Parameters
    ----------
    m1, m2 : floats
        Masses of the two masses.
    c_total : float
        Total damping to be distributed.
    k1, k2 : floats
        Fixed stiffnesses for the two springs.
    theta_0_1, theta_0_2 : floats
        Constant forcing offsets (may be zero).
    x0 : array-like of shape (4,)
        Initial condition [x1(0), v1(0), x2(0), v2(0)].
    t : array-like
        Time array of shape (N,).
    alpha_array : array-like
        The control array alpha(t) in [0,1], shape (N,).
    noise_array_1, noise_array_2 : array-like
        Noise signals for v1_dot and v2_dot, shape (N,).

    Returns
    -------
    X_true : ndarray of shape (N, 4)
        Simulated state trajectory.
    X_dot_true : ndarray of shape (N, 4)
        Approx. derivatives of the states at each time step.
    """
    N = len(t)
    X_true = np.zeros((N, 4))
    X_true[0] = x0

    # Helper for the ODE
    def dynamics(t_val, state, c1_val, c2_val, n1, n2):
        x1, v1, x2, v2 = state

        dx1 = v1
        dx2 = v2

        dv1 = (
            theta_0_1
            - c1_val*v1
            - k1*x1
            - c2_val*(v1 - v2)
            - k2*(x1 - x2)
        ) / m1 + n1

        dv2 = (
            theta_0_2
            - c2_val*(v2 - v1)
            - k2*(x2 - x1)
        ) / m2 + n2

        return [dx1, dv1, dx2, dv2]

    # Step-by-step integration using solve_ivp
    for i in range(N - 1):
        dt_span = [t[i], t[i+1]]
        t_eval = [t[i+1]]

        alpha_i = alpha_array[i]
        c1_i = alpha_i * c_total
        c2_i = (1.0 - alpha_i) * c_total

        sol = solve_ivp(
            dynamics,
            dt_span,
            X_true[i],
            args=(c1_i, c2_i, noise_array_1[i], noise_array_2[i]),
            method='RK45',
            t_eval=t_eval
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed at step {i}: {sol.message}")
        X_true[i+1] = sol.y[:, -1]

    # Compute approximate X_dot
    X_dot_true = np.zeros_like(X_true)
    for i in range(N - 1):
        x1, v1, x2, v2 = X_true[i]
        alpha_i = alpha_array[i]
        c1_i = alpha_i * c_total
        c2_i = (1.0 - alpha_i) * c_total

        dx1 = v1
        dv1 = (
            theta_0_1
            - c1_i*v1
            - k1*x1
            - c2_i*(v1 - v2)
            - k2*(x1 - x2)
        ) / m1 + noise_array_1[i]

        dx2 = v2
        dv2 = (
            theta_0_2
            - c2_i*(v2 - v1)
            - k2*(x2 - x1)
        ) / m2 + noise_array_2[i]

        X_dot_true[i] = [dx1, dv1, dx2, dv2]

    # For the final point, replicate the second to last derivative
    X_dot_true[-1] = X_dot_true[-2]

    return X_true, X_dot_true


def compute_true_coeff_v1dot(feature, m1, c_total, k1, k2, theta_0_1):
    """
    For v1_dot = [ theta_0_1
                   - alpha*c_total*v1
                   - k1*x1
                   - (1-alpha)*c_total*(v1 - v2)
                   - k2*(x1 - x2)
                 ] / m1

    Library naming:
       "x0" => x1
       "x1" => v1
       "x2" => x2
       "x3" => v2
       "u0" => alpha
       etc.
    """
    if feature == "1":
        return theta_0_1 / m1
    elif feature == "x0":  # x1
        return - (k1 + k2) / m1
    elif feature == "x1":  # v1
        # Summation: - alpha*c_total*v1 - (1-alpha)*c_total*v1 => -c_total*v1
        return - c_total / m1
    elif feature == "x2":  # x2
        return + k2 / m1
    elif feature == "x3":  # v2
        # from -(1-alpha)*c_total * (-v2) => + (1-alpha)*c_total*v2 => c_total*v2 - alpha*c_total*v2
        # => pure v2 => + c_total, cross v2*alpha => - c_total => see next
        return + c_total / m1
    elif feature == "x3 u0":  # v2 * alpha
        return - c_total / m1
    # All other terms => 0
    return 0.0


def compute_true_coeff_v2dot(feature, m2, c_total, k2, theta_0_2):
    """
    v2_dot = [ theta_0_2
               - (1-alpha)*c_total*(v2 - v1)
               - k2*(x2 - x1)
             ] / m2
    = 1/m2 [ theta_0_2
             + (1-alpha)*c_total*v1
             - (1-alpha)*c_total*v2
             - k2*x2 + k2*x1
           ]
    """
    if feature == "1":
        return theta_0_2 / m2
    elif feature == "x0":  # x1
        return + k2 / m2
    elif feature == "x1":  # v1
        # (1-alpha)*c_total*v1 => c_total*v1 - alpha*c_total*v1 => pure v1 => + c_total
        return + c_total / m2
    elif feature == "x2":  # x2
        return - k2 / m2
    elif feature == "x3":  # v2
        # -(1-alpha)*c_total*v2 => - c_total*v2 + alpha*c_total*v2 => pure v2 => - c_total
        return - c_total / m2
    elif feature == "x1 u0":  # v1*alpha
        # - alpha*c_total*v1 => => - c_total
        return - c_total / m2
    elif feature == "x3 u0":  # v2*alpha
        # + alpha*c_total*v2 => => + c_total
        return + c_total / m2
    return 0.0


def compute_true_coeffs(
    m1, m2, c1, c2, k1, k2,
    theta_0_1, theta_0_2,
    sigma_epsilon_1, sigma_epsilon_2,
    feature_names
):
    """
    For a known 2DOF system with two distinct control inputs (u0, u1),
    compute the "true" parameter vector.

    v1_dot = theta_0_1
             - (c1 + c2)/m1 * v1
             + c2/m1       * v2
             - (k1 + k2)/m1* x1
             + k2/m1       * x2
             + 1/m1        * u0    (no dependence on u1 for v1_dot)

    v2_dot = theta_0_2
             + c2/m2       * v1
             - c2/m2       * v2
             + k2/m2       * x1
             - k2/m2       * x2
             + 1/m2        * u1    (no dependence on u0 for v2_dot)

    The library is typically:
       1 (bias),
       x1, v1, x2, v2,
       u0, u1,
       plus possibly polynomial cross-terms if poly_degree>1, etc.

    We'll set the correct linear coefficients, and 0 for cross-terms.
    """
    # Number of features minus 1 for the bias
    # (if your library has 7 features: 1, x1, v1, x2, v2, u0, u1)
    # then each equation has 6 coefficients to fill.
    n_features_minus_bias = len(feature_names) - 1
    coeffs_v1 = np.zeros(n_features_minus_bias)
    coeffs_v2 = np.zeros(n_features_minus_bias)

    # Suppose your feature_names (excluding bias) appear in the order:
    # [x1, v1, x2, v2, u0, u1, ... maybe more ...]
    # We'll assume the first 4 are the states, next are inputs.
    # Adjust indexing if your library’s ordering differs.

    # v1_dot
    # x1
    coeffs_v1[0] = -(k1 + k2)/m1
    # v1
    coeffs_v1[1] = -(c1 + c2)/m1
    # x2
    coeffs_v1[2] = k2/m1
    # v2
    coeffs_v1[3] = c2/m1
    # u0
    coeffs_v1[4] = 1.0/m1
    # u1 => 0 in v1_dot, if it appears, set it to 0
    if len(coeffs_v1) > 5:
        coeffs_v1[5] = 0.0

    # v2_dot
    # x1
    coeffs_v2[0] = k2/m2
    # v1
    coeffs_v2[1] = c2/m2
    # x2
    coeffs_v2[2] = -(k2/m2)
    # v2
    coeffs_v2[3] = -(c2/m2)
    # u0 => 0 in v2_dot
    coeffs_v2[4] = 0.0
    # u1 => 1/m2 in v2_dot if present
    if len(coeffs_v2) > 5:
        coeffs_v2[5] = 1.0/m2

    # Final parameter vector = [theta_0_1, coeffs_v1..., theta_0_2, coeffs_v2...]
    true_params = (
        [theta_0_1]
        + list(coeffs_v1)
        + [theta_0_2]
        + list(coeffs_v2)
    )
    return true_params
