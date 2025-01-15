"2DOF feature definitions"

import numpy as np

def get_feature_names():
    """
    Returns the list of feature names used by the model for the 2DOF system.
    For simplicity, we mimic the pattern of the SDOF but now with x1, v1, x2, v2.
    """
    feature_names = [
        'x1', 'v1', 'x2', 'v2',
        'x1^2', 'v1^2', 'x2^2', 'v2^2',
        # 'x1*v1', 'x1*x2', 'x1*v2', 'v1*x2', 'v1*v2', 'x2*v2',
        'u'
    ]
    return feature_names

def compute_features_vectorized(X, U):
    """
    Compute the feature matrix for all samples.
    X: (n_samples, 4) with columns [x1, v1, x2, v2]
    U: (n_samples,) control input applied to the second mass.
    """
    x1 = X[:, 0]
    v1 = X[:, 1]
    x2 = X[:, 2]
    v2 = X[:, 3]
    u_val = U if U is not None else np.zeros_like(x1)
    
    Theta = np.column_stack([
        x1,
        v1,
        x2,
        v2,
        x1**2,
        v1**2,
        x2**2,
        v2**2,
        x1*v1,
        x1*x2,
        x1*v2,
        v1*x2,
        v1*v2,
        x2*v2,
        u_val
    ])
    return Theta

def build_library(X, U=None):
    """
    Build the library of candidate features from the 2DOF states and control input.
    """
    feature_names = get_feature_names()
    Theta = compute_features_vectorized(X, U)
    return Theta, feature_names

def compute_true_coeffs(m1, m2, c1, c2, k1, k2, theta_0_1, theta_0_2, sigma_epsilon_1, sigma_epsilon_2, feature_names):
    """
    Compute the "true" coefficients for both equations if known.
    For simplicity, assume a linear system:
    v1_dot = theta_0_1 + (- (c1+c2)/m1)*v1 + (c2/m1)*v2 + (- (k1+k2)/m1)*x1 + (k2/m1)*x2
    v2_dot = theta_0_2 + (c2/m2)*v1 + (-c2/m2)*v2 + (k2/m2)*x1 + (-k2/m2)*x2 + (1/m2)*u

    Nonlinear terms' coefficients = 0, just as in the SDOF example.
    This is an example; adjust as needed.
    """

    # Initialize all to zero
    coeffs_1 = np.zeros(len(feature_names))
    coeffs_2 = np.zeros(len(feature_names))

    # Identify indices
    fname = feature_names
    idx_x1 = fname.index('x1')
    idx_v1 = fname.index('v1')
    idx_x2 = fname.index('x2')
    idx_v2 = fname.index('v2')
    idx_u  = fname.index('u')
    
    # Equation for v1_dot
    coeffs_1[idx_x1] = -(k1+k2)/m1
    coeffs_1[idx_v1] = -(c1+c2)/m1
    coeffs_1[idx_x2] = (k2/m1)
    coeffs_1[idx_v2] = (c2/m1)
    # No u term in v1_dot
    # All nonlinear terms zero
    
    # Equation for v2_dot
    coeffs_2[idx_x1] = (k2/m2)
    coeffs_2[idx_v1] = (c2/m2)
    coeffs_2[idx_x2] = -(k2/m2)
    coeffs_2[idx_v2] = -(c2/m2)
    coeffs_2[idx_u]  = 1/m2
    # Nonlinear terms zero

    # Combine into a single parameter vector for convenience:
    # [theta_0_1, coeffs_1..., theta_0_2, coeffs_2..., sigma_epsilon_1, sigma_epsilon_2]
    # Insert theta_0 and sigma terms at the end
    true_params = [theta_0_1] + list(coeffs_1) + [theta_0_2] + list(coeffs_2) + [sigma_epsilon_1, sigma_epsilon_2]
    return true_params

# TwoDOF_feature_definitions.py

import numpy as np

###############################################################################


###############################################################################
# 2) Build the feature matrix (Theta) by interpreting each feature name
###############################################################################
# def compute_features_vectorized(
#     X, 
#     U, 
#     feature_names
# ):
#     """
#     Compute the feature matrix for each sample in X, matching the columns
#     to the order of 'feature_names'.

#     Parameters
#     ----------
#     X : (n_samples, 4) array
#         [x1, v1, x2, v2].
#     U : (n_samples,) or None
#         Control input for second mass.
#     feature_names : list of str
#         The feature names in the order we want to build columns.

#     Returns
#     -------
#     Theta : (n_samples, n_features) array
#         Feature matrix.
#     """
#     # Unpack states
#     x1 = X[:, 0]
#     v1 = X[:, 1]
#     x2 = X[:, 2]
#     v2 = X[:, 3]

#     # If U is None, treat as zero input
#     if U is None:
#         U = np.zeros_like(x1)
#     elif U.ndim == 2 and U.shape[1] == 1:
#         U = U.flatten()

#     cols = []

#     for fname in feature_names:
#         if fname == "x1":
#             cols.append(x1)
#         elif fname == "v1":
#             cols.append(v1)
#         elif fname == "x2":
#             cols.append(x2)
#         elif fname == "v2":
#             cols.append(v2)
#         elif fname == "u":
#             cols.append(U)

#         elif fname == "x1^2":
#             cols.append(x1**2)
#         elif fname == "v1^2":
#             cols.append(v1**2)
#         elif fname == "x2^2":
#             cols.append(x2**2)
#         elif fname == "v2^2":
#             cols.append(v2**2)
#         elif fname == "u^2":
#             cols.append(U**2)

#         elif "*" in fname:
#             # e.g. "x1*v1", "x1*x2"
#             left, right = fname.split("*")
#             def get_var(var_name):
#                 if var_name == "x1":
#                     return x1
#                 elif var_name == "v1":
#                     return v1
#                 elif var_name == "x2":
#                     return x2
#                 elif var_name == "v2":
#                     return v2
#                 else:
#                     raise ValueError(f"Unknown variable: {var_name}")
#             cols.append(get_var(left) * get_var(right))

#         else:
#             raise ValueError(f"Unrecognized feature name: {fname}")

#     Theta = np.column_stack(cols)
#     return Theta

# ###############################################################################
# # 3) Build the library in one shot
# ###############################################################################
# def build_library(
#     X, 
#     U=None,
#     include_linear=True,
#     include_quadratic=True,
#     include_interactions=False
# ):
#     """
#     Wrapper to create the final feature matrix (Theta) plus the corresponding 
#     feature names, for the 2DOF system, given the chosen flags.

#     Returns
#     -------
#     Theta : (n_samples, n_features) array
#     feature_names : list of str
#     """
#     feature_names = get_feature_names(
#         include_linear=include_linear,
#         include_quadratic=include_quadratic,
#         include_interactions=include_interactions
#     )
#     Theta = compute_features_vectorized(X, U, feature_names)
#     return Theta, feature_names

###############################################################################
# 4) Compute "true" coefficients based on the same feature names
###############################################################################
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
