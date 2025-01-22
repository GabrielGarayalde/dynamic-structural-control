import numpy as np
from scipy.integrate import solve_ivp

def simulate_true_varying_stiffness(
    m1, m2,
    c1, c2,
    k_total,
    x0, t,
    alpha_array,
    noise_array_1,
    noise_array_2,
    forcing_freq=None
):
    """
    Simulate a 2DOF system with total stiffness = k_total, distributed by alpha(t):
        k1(t) = alpha(t)*k_total
        k2(t) = (1 - alpha(t))*k_total.

    Optionally, you can include a sinusoidal forcing on mass 1 if forcing_freq is provided:
        F(t) = sin(forcing_freq * t).

    States: X = [x1, v1, x2, v2].

    Equations of motion:
      x1_dot = v1
      v1_dot = [ - c1*v1
                 - k1(t)*x1
                 - c2*(v1 - v2)
                 - k2(t)*(x1 - x2)
                 + ( optional sinusoid if forcing_freq is not None )
               ] / m1 + noise_1

      x2_dot = v2
      v2_dot = [ - c2*(v2 - v1)
                 - k2(t)*(x2 - x1)
               ] / m2 + noise_2
    """
    N = len(t)
    X_true = np.zeros((N, 4))
    X_true[0] = x0

    def dynamics(t_val, state, k1_val, k2_val, n1, n2):
        x1, v1, x2, v2 = state

        # Optional external forcing on mass 1
        if forcing_freq is not None:
            F = np.sin(forcing_freq * t_val)
        else:
            F = 0.0

        dx1 = v1
        dv1 = (
            - c1*v1
            - k1_val*x1
            - c2*(v1 - v2)
            - k2_val*(x1 - x2)
            + F
        ) / m1 + n1

        dx2 = v2
        dv2 = (
            - c2*(v2 - v1)
            - k2_val*(x2 - x1)
        ) / m2 + n2

        return [dx1, dv1, dx2, dv2]

    # Step-by-step integration using solve_ivp
    for i in range(N - 1):
        dt_span = [t[i], t[i+1]]
        t_eval = [t[i+1]]

        alpha_i = alpha_array[i]
        k1_i = alpha_i * k_total
        k2_i = (1.0 - alpha_i) * k_total

        sol = solve_ivp(
            dynamics,
            dt_span,
            X_true[i],
            args=(k1_i, k2_i, noise_array_1[i], noise_array_2[i]),
            method='RK45',
            t_eval=t_eval
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed at step {i}: {sol.message}")
        X_true[i+1] = sol.y[:, -1]

    # Compute approximate derivatives
    X_dot_true = np.zeros_like(X_true)
    for i in range(N - 1):
        x1, v1, x2, v2 = X_true[i]
        alpha_i = alpha_array[i]
        k1_i = alpha_i * k_total
        k2_i = (1.0 - alpha_i) * k_total

        if forcing_freq is not None:
            F = np.sin(forcing_freq * t[i])
        else:
            F = 0.0

        dx1 = v1
        dv1 = (
            - c1*v1
            - k1_i*x1
            - c2*(v1 - v2)
            - k2_i*(x1 - x2)
            + F
        ) / m1 + noise_array_1[i]

        dx2 = v2
        dv2 = (
            - c2*(v2 - v1)
            - k2_i*(x2 - x1)
        ) / m2 + noise_array_2[i]

        X_dot_true[i] = [dx1, dv1, dx2, dv2]

    X_dot_true[-1] = X_dot_true[-2]

    return X_true, X_dot_true


###############################################################################
# TRUE COEFFICIENTS: Single-control alpha(t) scenario
###############################################################################
def true_coefficient_v1dot(feature, m1, c1, c2, k_total, theta_0=0.0):
    """
    For the 2DOF system with alpha(t)*k_total on the first spring,
    (1-alpha)*k_total on the second spring.

    The derived "true" formula for v1_dot is:
      v1_dot = [ theta_0
                 - (c1 + c2)*v1
                 + c2*v2
                 - k_total*x1
                 + k_total*x2
                 - alpha*k_total*x2
               ] / m1

    Summarizing nonzero terms if the library has features like:
       1 ("bias")             => theta_0 / m1
       x0 (x1)                => -k_total / m1
       x1 (v1)                => -(c1 + c2)/m1
       x2 (x2)                => +k_total / m1
       x3 (v2)                => +c2 / m1
       (u0) = alpha           => 0
       (x2 * alpha) => x2*u0  => -k_total / m1
       ... all else => 0
    """
    if feature == "1":
        return theta_0 / m1
    elif feature == "x0":  # x1
        return -k_total / m1
    elif feature == "x1":  # v1
        return -(c1 + c2) / m1
    elif feature == "x2":  # x2
        return k_total / m1
    elif feature == "x3":  # v2
        return c2 / m1
    elif feature == "u0":  # alpha alone
        return 0.0
    elif feature == "x0 u0":  # x1*alpha
        return 0.0
    elif feature == "x1 u0":  # v1*alpha
        return 0.0
    elif feature == "x2 u0":  # x2*alpha
        return -k_total / m1
    elif feature == "x3 u0":  # v2*alpha
        return 0.0
    return 0.0  # cross-terms or higher-degree polynomials => 0


def true_coefficient_v2dot(feature, m2, c1, c2, k_total, theta_0=0.0):
    """
    The "true" formula for v2_dot is:

      v2_dot = [ theta_0
                 + c2*v1
                 - c2*v2
                 + (1 - alpha)*k_total*(x1 - x2)
               ] / m2

    Expand (1 - alpha)*k_total*(x1 - x2):
      = k_total*x1 - alpha*k_total*x1 - k_total*x2 + alpha*k_total*x2

    Summarizing nonzero terms if the library has:
      1 => theta_0 / m2
      x0 => +k_total / m2
      x1 => + c2 / m2
      x2 => -k_total / m2
      x3 => - c2 / m2
      (x0*u0) = x1*alpha => -k_total / m2
      (x2*u0) = x2*alpha => +k_total / m2
      all else => 0
    """
    if feature == "1":
        return theta_0 / m2
    elif feature == "x0":  # x1
        return k_total / m2
    elif feature == "x1":  # v1
        return c2 / m2
    elif feature == "x2":  # x2
        return -k_total / m2
    elif feature == "x3":  # v2
        return -c2 / m2
    elif feature == "u0":  # alpha alone
        return 0.0
    elif feature == "x0 u0":  # x1*alpha
        return -k_total / m2
    elif feature == "x1 u0":  # v1*alpha
        return 0.0
    elif feature == "x2 u0":  # x2*alpha
        return k_total / m2
    elif feature == "x3 u0":  # v2*alpha
        return 0.0
    return 0.0


def compute_true_coeffs_varying_stiffness(
    feature_names,
    m1, m2,
    c1, c2,
    k_total,
    theta_0_1=0.0,
    theta_0_2=0.0
):
    """
    Build arrays of "true" coefficients for v1_dot and v2_dot
    based on the known expansions for a single-control alpha(t) system.

    We do a simple loop over feature_names for v1_dot, calling true_coefficient_v1dot,
    and similarly for v2_dot calling true_coefficient_v2dot.

    Return: (true_v1, true_v2)
      each is a list of length == len(feature_names),
      corresponding to the same columns as model.get_feature_names().
    """
    true_v1 = []
    true_v2 = []
    for feat in feature_names:
        # For v1_dot
        coeff_v1 = true_coefficient_v1dot(feat, m1, c1, c2, k_total, theta_0_1)
        # For v2_dot
        coeff_v2 = true_coefficient_v2dot(feat, m2, c1, c2, k_total, theta_0_2)

        true_v1.append(coeff_v1)
        true_v2.append(coeff_v2)

    return np.array(true_v1), np.array(true_v2)
