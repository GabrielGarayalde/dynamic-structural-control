# simulate_2dof_nonlinear.py

import numpy as np
from scipy.integrate import solve_ivp

import numpy as np
from scipy.integrate import solve_ivp

def simulate_true(
    m1, m2, c1, c2, k1, k2,
    alpha1, alpha2,
    theta_0_1, theta_0_2,
    x0, t, U_array,
    noise_array_1, noise_array_2
):
    """
    Simulate the 2DOF Duffing system over [t[0], t[-1]] in one shot.
    We'll interpolate U(t) and noise(t) inside the dynamics.
    """
    # Interpolate your discrete arrays:
    def U(t_query):
        # E.g. use np.interp for simple 1D interpolation
        return np.interp(t_query, t, U_array)
    
    def noise1(t_query):
        return np.interp(t_query, t, noise_array_1)
    
    def noise2(t_query):
        return np.interp(t_query, t, noise_array_2)

    def dynamics(t_val, state):
        x1, v1, x2, v2 = state
        u_val = U(t_val)
        n1 = noise1(t_val)
        n2 = noise2(t_val)
        
        dx1 = v1
        dx2 = v2
        dv1 = (
            theta_0_1
            - c1*v1
            - k1*x1
            - c2*(v1 - v2)
            - k2*(x1 - x2)
            + alpha1*(x1**2)
        )/m1 + n1

        dv2 = (
            theta_0_2
            + u_val
            - c2*(v2 - v1)
            - k2*(x2 - x1)
            + alpha2*(x2**2)
        )/m2 + n2

        return [dx1, dv1, dx2, dv2]

    # Call solve_ivp once over the entire time span
    sol = solve_ivp(
        fun=dynamics,
        t_span=(t[0], t[-1]),
        y0=x0,
        t_eval=t,        # we want the solution at each point in t
        method='RK45',    # or 'LSODA' or 'BDF' if stiff
        rtol=1e-6,        # you can try relaxing this
        atol=1e-8         # or this
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Extract solution
    X_true = sol.y.T  # shape (len(t), 4)

    # We can compute derivatives from the final continuous solution:
    X_dot_true = np.zeros_like(X_true)
    for i in range(len(t)):
        # evaluate dynamics at each time
        dx = dynamics(t[i], X_true[i])
        X_dot_true[i] = dx

    return X_true, X_dot_true





def compute_true_coeffs(
    m1, m2, c1, c2, k1, k2,
    theta_0_1, theta_0_2,
    sigma_epsilon_1, sigma_epsilon_2,
    feature_names,
    alpha1=0.0,    # Duffing for x0^3 in the v1_dot eqn
    alpha2=0.0,    # Duffing for x2^3 in the v2_dot eqn
):
    """
    Compute the "true" parameter vector for a 2DOF system, which may
    optionally include Duffing-type cubic terms (alpha1*x0^3, alpha2*x2^3).

    We assume:
      v1_dot = theta_0_1
               - (c1 + c2)/m1 * v1
               + c2/m1 * v2
               - (k1 + k2)/m1 * x1
               + k2/m1 * x2
               + alpha1 * x1^3  (if present in library)

      v2_dot = theta_0_2
               + c2/m2 * v1
               - c2/m2 * v2
               + k2/m2 * x1
               - k2/m2 * x2
               + (1/m2)*u
               + alpha2 * x2^3  (if present in library)

    Parameters
    ----------
    feature_names : list of str
        The library features. Typically includes ['1', 'x0', 'x1', 'x2', 'x3',
        'x0^2', 'x0*x1', ... etc.].
        *The first entry is often '1' (the bias).*
    
    alpha1 : float
        Coefficient for x0^3 in v1_dot (x0 = x1 in your physical notation).
    alpha2 : float
        Coefficient for x2^3 in v2_dot (x2 = x2 in your physical notation).

    Returns
    -------
    true_params : list of floats
        [theta_0_1, <coeffs_v1>, theta_0_2, <coeffs_v2>].
    """
    # We skip the last 2 for sigma_epsilon_1, sigma_epsilon_2, 
    # but sometimes you might want to append them at the end.
    # For now we won't place them in the array automatically,
    # or you can place them if your use-case requires it.

    # ----------------------------------------------------------------------
    # 1) Initialize coefficient arrays for v1_dot, v2_dot
    #    NOTE: We subtract 1 because feature_names[0] = '1' is the bias term,
    #    and typically when we do .coefficients_ in PySINDy, 
    #    the columns match the library EXCEPT that row 0 is for the bias.
    # ----------------------------------------------------------------------
    n_features_no_bias = len(feature_names) - 1
    coeffs_v1 = np.zeros(n_features_no_bias)
    coeffs_v2 = np.zeros(n_features_no_bias)

    # ----------------------------------------------------------------------
    # 2) Fill in the known linear physical terms
    #    Suppose your library for v1_dot includes: x0, x1, x2, x3, u, x0^2, ...
    #    You just need to locate them by name.
    # ----------------------------------------------------------------------
    # For v1_dot:
    #   - (k1 + k2)/m1 * x0  => feature_name 'x0'
    #   - (c1 + c2)/m1 * x1  => 'x1'
    #   + (k2/m1) * x2       => 'x2'
    #   + (c2/m1) * x3       => 'x3'
    #   ... etc.

    # A little helper function to place a value in coeffs_v1 or coeffs_v2
    # if the feature name is present.
    def set_coeff(array, feat_name, value):
        """Utility to place `value` in `array` at the index of `feat_name` (if found)."""
        try:
            idx = feature_names.index(feat_name)
            # subtract 1 to account for the bias being feature_names[0]
            array[idx - 1] = value
        except ValueError:
            pass  # e.g. if 'x0' is not in the library, do nothing.

    # v1_dot
    set_coeff(coeffs_v1, 'x0', -(k1 + k2)/m1)  # x1 => x0 in code
    set_coeff(coeffs_v1, 'x1', -(c1 + c2)/m1)  # v1 => x1 in code
    set_coeff(coeffs_v1, 'x2',  (k2/m1))       # x2 => x2 in code
    set_coeff(coeffs_v1, 'x3',  (c2/m1))       # v2 => x3 in code
    # If there's a 'u' in v1_dot library, it is 0 for v1_dot
    # set_coeff(coeffs_v1, 'u', 0.0)  # not strictly needed

    # v2_dot
    set_coeff(coeffs_v2, 'x0',  (k2/m2))       # x1 => x0 in code
    set_coeff(coeffs_v2, 'x1',  (c2/m2))       # v1 => x1 in code
    set_coeff(coeffs_v2, 'x2', -(k2/m2))       # x2 => x2 in code
    set_coeff(coeffs_v2, 'x3', -(c2/m2))       # v2 => x3 in code
    set_coeff(coeffs_v2, 'u0',   1.0/m2)        # external forcing on second mass

    # ----------------------------------------------------------------------
    # 3) Place Duffing coefficients (alpha1 -> x0^3 in v1_dot, alpha2 -> x2^3 in v2_dot)
    # ----------------------------------------------------------------------
    # x0^3 is the "x1^3" physically, but in the code x0 is x1, so we look for 'x0^3'.
    if alpha1 != 0.0:
        set_coeff(coeffs_v1, 'x0^2', alpha1)

    # x2^3 is the "x2^3" physically, so we look for 'x2^3'.
    if alpha2 != 0.0:
        set_coeff(coeffs_v2, 'x2^2', alpha2)

    # ----------------------------------------------------------------------
    # 4) Construct final parameter vector
    #    (theta_0_1, v1_dot coefficients..., theta_0_2, v2_dot coefficients...)
    # ----------------------------------------------------------------------
    true_params = [theta_0_1] + list(coeffs_v1) + [theta_0_2] + list(coeffs_v2)

    # OPTIONAL: if you want to append sigma_epsilon_1 and sigma_epsilon_2 at the end:
    #   true_params = true_params + [sigma_epsilon_1, sigma_epsilon_2]
    true_params = np.round(true_params, 3).tolist()

    return true_params
