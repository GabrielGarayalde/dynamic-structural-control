import numpy as np
from scipy.integrate import solve_ivp

def simulate_true(
    m1, m2,
    c1, c2,
    k1, k2,
    alpha1, alpha2,
    theta_0_1, theta_0_2,
    x0, t, U,
    noise_array_1, noise_array_2
):
    """
    Simulate a 2DOF 'Duffing-style' system with:
      v1_dot = [theta_0_1 + u1(t)]
                - c1*v1 - k1*x1 - c2*(v1 - v2) - k2*(x1 - x2)
                - alpha1 * x1^3
                + [optional cubic coupling terms if you like]
                all divided by m1
              + noise_1

      v2_dot = [theta_0_2 + u2(t)]
                - c2*(v2 - v1) - k2*(x2 - x1)
                - alpha2 * x2^3
                all divided by m2
              + noise_2

    States: X = [x1, v1, x2, v2]
    """

    X_true = np.zeros((len(t), 4))
    X_true[0] = x0

    def dynamics(t_val, state, u_val_1, u_val_2, n1, n2):
        x1, v1, x2, v2 = state
        dx1 = v1
        dx2 = v2

        # Example cubic terms:
        cubic_term_1 = alpha1 * (x1**3)
        # If you want coupling cubic, you could do alpha_c * ((x1 - x2)**3), etc.
        cubic_term_2 = alpha2 * (x2**3)

        dv1 = (
            theta_0_1 + u_val_1
            - c1*v1
            - k1*x1
            - c2*(v1 - v2)
            - k2*(x1 - x2)
            - cubic_term_1
        ) / m1 + n1

        dv2 = (
            theta_0_2 + u_val_2
            - c2*(v2 - v1)
            - k2*(x2 - x1)
            - cubic_term_2
        ) / m2 + n2

        return [dx1, dv1, dx2, dv2]

    for i in range(len(t) - 1):
        dt_span = [t[i], t[i+1]]
        t_eval = [t[i+1]]

        # Controls at time-step i
        u1_i = U[i, 0]
        u2_i = U[i, 1]

        sol = solve_ivp(
            dynamics,
            dt_span,
            X_true[i],
            args=(u1_i, u2_i, noise_array_1[i], noise_array_2[i]),
            method='RK45',
            t_eval=t_eval
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed at step {i}: {sol.message}")
        X_true[i+1] = sol.y[:, -1]

    # Compute derivatives X_dot_true for each time-step
    X_dot_true = np.zeros_like(X_true)
    for i in range(len(t) - 1):
        x1, v1, x2, v2 = X_true[i]
        dx1 = v1
        dx2 = v2

        cubic_term_1 = alpha1 * (x1**3)
        cubic_term_2 = alpha2 * (x2**3)

        dv1 = (
            theta_0_1 + U[i, 0]
            - c1*v1
            - k1*x1
            - c2*(v1 - v2)
            - k2*(x1 - x2)
            - cubic_term_1
        ) / m1 + noise_array_1[i]

        dv2 = (
            theta_0_2 + U[i, 1]
            - c2*(v2 - v1)
            - k2*(x2 - x1)
            - cubic_term_2
        ) / m2 + noise_array_2[i]

        X_dot_true[i] = [dx1, dv1, dx2, dv2]

    # Last derivative can be a copy of second-to-last:
    X_dot_true[-1] = X_dot_true[-2]

    return X_true, X_dot_true

def compute_true_coeffs(
    m1, m2,
    c1, c2,
    k1, k2,
    alpha1, alpha2,
    theta_0_1, theta_0_2,
    feature_names
):
    """
    Return [theta_0_1, <coeffs for v1_dot>, theta_0_2, <coeffs for v2_dot>].
    Now includes cubic x1^3 and x2^3 terms if those appear in feature_names.
    """
    # e.g. feature_names might be:
    # ["1", "x0", "x1", "x2", "x3", "x0^2", "x0^3", "x2^3", "u0", "u1", ...]
    #
    # where x0->x1, x1->v1, x2->x2, x3->v2 in your code.

    # Make arrays of zeros for v1_dot & v2_dot
    coeffs_v1 = np.zeros(len(feature_names) - 1)  # minus 1 if '1' is the bias
    coeffs_v2 = np.zeros(len(feature_names) - 1)

    # Example mapping:
    # index_of('x0') => i_x1
    # index_of('x0^3') => i_x1_cubed
    # etc.
    i_bias = 0  # feature_names[0] = "1"
    i_x1   = feature_names.index("x0")   - 1
    i_v1   = feature_names.index("x1")   - 1
    i_x2   = feature_names.index("x2")   - 1
    i_v2   = feature_names.index("x3")   - 1
    i_x1_3 = feature_names.index("x0^3") - 1 if "x0^3" in feature_names else None
    i_x2_3 = feature_names.index("x2^3") - 1 if "x2^3" in feature_names else None
    i_u0   = feature_names.index("u0")   - 1 if "u0"   in feature_names else None
    i_u1   = feature_names.index("u1")   - 1 if "u1"   in feature_names else None

    # Now fill them in for v1_dot:
    #   v1_dot = theta_0_1 + [stuff]/m1
    # e.g. - (c1 + c2)*v1, - (k1 + k2)*x1, etc. minus alpha1*x1^3
    # example:
    if i_x1   is not None: coeffs_v1[i_x1]   = -(k1 + k2)/m1
    if i_v1   is not None: coeffs_v1[i_v1]   = -(c1 + c2)/m1
    if i_x2   is not None: coeffs_v1[i_x2]   = (k2 / m1)
    if i_v2   is not None: coeffs_v1[i_v2]   = (c2 / m1)
    if i_x1_3 is not None: coeffs_v1[i_x1_3] = -(alpha1 / m1)
    if i_u0   is not None: coeffs_v1[i_u0]   = 1.0 / m1
    if i_u1   is not None: coeffs_v1[i_u1]   = 0.0  # no direct effect on v1

    # v2_dot:
    #   v2_dot = theta_0_2 + [stuff]/m2
    # e.g. -alpha2*x2^3, etc.
    if i_x1   is not None: coeffs_v2[i_x1]   = k2/m2
    if i_v1   is not None: coeffs_v2[i_v1]   = c2/m2
    if i_x2   is not None: coeffs_v2[i_x2]   = -(k2 / m2)
    if i_v2   is not None: coeffs_v2[i_v2]   = -(c2 / m2)
    if i_x2_3 is not None: coeffs_v2[i_x2_3] = -(alpha2 / m2)
    if i_u0   is not None: coeffs_v2[i_u0]   = 0.0
    if i_u1   is not None: coeffs_v2[i_u1]   = 1.0 / m2

    true_params = (
        [theta_0_1]
        + list(coeffs_v1)
        + [theta_0_2]
        + list(coeffs_v2)
    )
    return true_params
