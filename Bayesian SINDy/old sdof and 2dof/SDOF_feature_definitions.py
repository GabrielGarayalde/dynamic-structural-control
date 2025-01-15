import numpy as np

# ------------------------------
# 1. Modular Feature Definitions
# ------------------------------

def get_feature_names():
    """
    Returns the list of feature names used by the model.
    Edit this function to add/remove/comment out features in one place.
    """
    # Current chosen features:
    # x, v, x^2, v^2, x*v, u
    feature_names = ['x', 'v', 
                     'x^2', 'v^2',
                     'x*v',
                     'u']
    return feature_names

def compute_features_vectorized(X, U):
    """
    Compute the feature matrix for all samples at once.
    X: (n_samples, 2) with columns [x, v]
    U: (n_samples,) control input applied to the system.
    Returns:
        Theta: (n_samples, M) feature matrix
    """
    x = X[:, 0]
    v = X[:, 1]
    u_val = U if U is not None else np.zeros_like(x)
    
    # Compute features based on get_feature_names()
    Theta = np.column_stack([
        x,          # 'x'
        v,          # 'v'
        x**2,       # 'x^2'
        v**2,       # 'v^2'
        x * v,      # 'x*v'
        u_val       # 'u'
    ])
    return Theta

def build_library(X, U=None):
    """
    Build a library of candidate features from the state variables and control input u.
    X: (n_samples, 2) with columns [x, v]
    U: (n_samples,) control input applied to the system.
    """
    feature_names = get_feature_names()
    Theta = compute_features_vectorized(X, U)
    return Theta, feature_names

def compute_true_coeffs(m, k, c, theta_0, sigma_epsilon, feature_names):
    """
    Compute the true coefficients including the control input 'u'.
    Align this with the features in get_feature_names().
    Features: [x, v, x^2, v^2, x*v, u]
    """
    # Define coefficients based on structural dynamics
    # For v_dot = theta_0 + c_v_x * x + c_v_v * v + c_v_x2 * x^2 + c_v_v2 * v^2 + c_v_xv * x*v + c_v_u * u
    # Assuming that the true system is linear, set non-linear coefficients to zero

    # Coefficients for v_dot
    c_v_x = -k/m
    c_v_v = -c/m
    c_v_x2 = 0.0
    c_v_v2 = 0.0
    c_v_xv = 0.0
    c_v_u = 1.0/m

    # Order must match feature_names
    true_coeffs = [
        theta_0,       # theta_0
        c_v_x,         # c_v_x
        c_v_v,         # c_v_v
        c_v_x2,        # c_v_x2
        c_v_v2,        # c_v_v2
        c_v_xv,        # c_v_xv
        c_v_u,         # c_v_u
        sigma_epsilon  # sigma_epsilon
    ]
    return true_coeffs
