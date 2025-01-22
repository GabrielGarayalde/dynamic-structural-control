import numpy as np
from scipy.integrate import solve_ivp

def simulate_true_varying_damping(
    m1, m2,
    c_total,
    k1, k2,
    x0, t,
    alpha_array,
    noise_array_1,
    noise_array_2,
    forcing_freq=None
):
    """
    Simulate a 2DOF system where total damping c_total is split by alpha(t):
        c1(t) = alpha(t)*c_total
        c2(t) = (1 - alpha(t))*c_total

    The stiffnesses k1, k2 remain constant.

    Optionally apply a sinusoidal forcing on mass 1:
        F(t) = sin(forcing_freq * t).

    States: X = [x1, v1, x2, v2].
    """
    N = len(t)
    X_true = np.zeros((N, 4))
    X_true[0] = x0

    def dynamics(t_val, state, c1_val, c2_val, n1, n2):
        x1, v1, x2, v2 = state

        # Optional forcing on mass1
        if forcing_freq is not None:
            forcing_amp = 1.0  # or 2.0, or 10.0, etc.
            F = forcing_amp * np.sin(forcing_freq * t_val)
        else:
            F = 0.0

        dx1 = v1
        dv1 = (
            - c1_val*v1
            - k1*x1
            - c2_val*(v1 - v2)
            - k2*(x1 - x2)
            + F
        ) / m1 + n1

        dx2 = v2
        dv2 = (
            - c2_val*(v2 - v1)
            - k2*(x2 - x1)
        ) / m2 + n2

        return [dx1, dv1, dx2, dv2]

    # Step integration
    for i in range(N - 1):
        c1_i = alpha_array[i] * c_total
        c2_i = (1.0 - alpha_array[i]) * c_total

        dt_span = [t[i], t[i+1]]
        t_eval = [t[i+1]]

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

    # Approximate derivatives
    X_dot_true = np.zeros_like(X_true)
    for i in range(N - 1):
        c1_i = alpha_array[i] * c_total
        c2_i = (1.0 - alpha_array[i]) * c_total

        if forcing_freq is not None:
            F = np.sin(forcing_freq * t[i])
        else:
            F = 0.0

        x1, v1, x2, v2 = X_true[i]
        dx1 = v1
        dv1 = (
            - c1_i*v1
            - k1*x1
            - c2_i*(v1 - v2)
            - k2*(x1 - x2)
            + F
        ) / m1 + noise_array_1[i]

        dx2 = v2
        dv2 = (
            - c2_i*(v2 - v1)
            - k2*(x2 - x1)
        ) / m2 + noise_array_2[i]

        X_dot_true[i] = [dx1, dv1, dx2, dv2]

    X_dot_true[-1] = X_dot_true[-2]
    return X_true, X_dot_true


###############################################################################
# True Coeff Expansions
###############################################################################
def true_coefficient_v1dot(feature, m1, c_total, k1, k2, theta_0=0.0):
    """
    With c1=alpha*c_total, c2=(1-alpha)*c_total, plus constant k1, k2.
    The 'true' eqn for v1_dot is roughly:

      v1_dot = 1/m1 [ theta_0
                      - alpha*c_total*v1
                      - k1*x1
                      - (1-alpha)*c_total*(v1 - v2)
                      - k2*(x1 - x2)
                    ]

    => Expand (1-alpha)*c_total*(v1 - v2):
         = c_total*v1 - alpha*c_total*v1 - c_total*v2 + alpha*c_total*v2
       So we'll identify terms x1, v1, x2, v2, alpha*x2, etc.

    We'll match them to the standard PySINDy features: x0->x1, x1->v1, x2->x2, x3->v2, u0->alpha, ...
    """
    # Summarizing:
    # v1_dot = 1/m1 * [ theta_0
    #                   - c_total*v1
    #                   + c_total*v2
    #                   - alpha*c_total*v2
    #                   - k1*x1
    #                   - k2*x1
    #                   + k2*x2
    #                   + (maybe alpha pieces for x2?)
    #                 ]
    # Combine terms carefully. We'll assume the library includes e.g. 'x2*u0' => alpha*x2, etc.
    #
    # You can systematically work out each. For simplicity, let's do a partial example:
    if feature == "1":
        return theta_0 / m1
    elif feature == "x0":  # x1
        return -(k1 + k2)/m1
    elif feature == "x1":  # v1
        return - (c_total)/m1
    elif feature == "x2":  # x2
        return + (k2)/m1
    elif feature == "x3":  # v2
        return + (c_total)/m1
    elif feature == "u0":  # alpha
        # alpha alone => does not appear as a direct term
        return 0.0
    elif feature == "x3 u0":  # v2*alpha
        # from + c_total*v2 - alpha*c_total*v2 => that leftover is - c_total v2
        return - c_total/m1
    elif feature == "x2 u0":  # x2*alpha
        # Possibly if (1-alpha)*c_total*(something) => see expansions
        # Actually for x2, let's see: there's no direct alpha*x2 in that expansion except maybe partial from -(1-alpha)* ...
        # We'll skip. Or if we see a needed term => 0
        return 0.0
    # ...
    return 0.0


def true_coefficient_v2dot(feature, m2, c_total, k2, theta_0=0.0):
    """
    v2_dot = 1/m2 [ theta_0
                    - (1-alpha)*c_total*(v2 - v1)
                    - k2*(x2 - x1)
                  ]
    => expand => ...
    """
    if feature == "1":
        return theta_0 / m2
    elif feature == "x0":  # x1
        return +k2/m2
    elif feature == "x1":  # v1
        return + (c_total)/m2
    elif feature == "x2":  # x2
        return -k2/m2
    elif feature == "x3":  # v2
        return -(c_total)/m2
    elif feature == "u0":  # alpha alone
        return 0.0
    elif feature == "x1 u0":  # v1 * alpha
        return - c_total/m2
    elif feature == "x3 u0":  # v2 * alpha
        return + c_total/m2
    return 0.0


def compute_true_coeffs_varying_damping(
    feature_names,
    m1, m2,
    c_total,
    k1, k2,
    theta_0_1=0.0,
    theta_0_2=0.0
):
    """
    Build arrays of "true" coefficients for v1_dot and v2_dot
    based on expansions for single-control alpha(t) distributing damping.
    """
    true_v1 = []
    true_v2 = []
    for feat in feature_names:
        coeff_v1 = true_coefficient_v1dot(feat, m1, c_total, k1, k2, theta_0_1)
        coeff_v2 = true_coefficient_v2dot(feat, m2, c_total, k2, theta_0_2)
        true_v1.append(coeff_v1)
        true_v2.append(coeff_v2)

    return np.array(true_v1), np.array(true_v2)
