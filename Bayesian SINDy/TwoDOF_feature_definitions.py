"2DOF feature definitions"

import numpy as np

def get_feature_names():
    """
    Returns the list of feature names used by the model for the 2DOF system.
    For simplicity, we mimic the pattern of the SDOF but now with x1, v1, x2, v2.
    """
    feature_names = [
        'x1', 'v1', 'x2', 'v2',
        # 'x1^2', 'v1^2', 'x2^2', 'v2^2',
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
        # x1**2,
        # v1**2,
        # x2**2,
        # v2**2,
        # x1*v1,
        # x1*x2,
        # x1*v2,
        # v1*x2,
        # v1*v2,
        # x2*v2,
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
