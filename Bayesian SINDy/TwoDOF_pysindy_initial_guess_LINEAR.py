import pysindy as ps
import numpy as np

def get_initial_guess_from_pysindy(X, X_dot, U, t):
    """
    Use PySINDy to get an initial guess for the model parameters.
    Assumptions:
    - The Bayesian model expects features: [x1, v1, x2, v2, u] for v1_dot and v2_dot equations.
    - The parameter vector format: [theta_0_1, c_v1_x1, c_v1_v1, c_v1_x2, c_v1_v2, c_v1_u,
                                    theta_0_2, c_v2_x1, c_v2_v1, c_v2_x2, c_v2_v2, c_v2_u,
                                    sigma_epsilon_1, sigma_epsilon_2]
    - PySINDy may return duplicate features, so we handle that by deduplicating and averaging.

    Returns:
        initial_guess: np.array of shape (14,) corresponding to the parameter vector.
    """

    # Reshape U if necessary
    if U.ndim == 1:
        U = U.reshape(-1,1)

    # Create a simple polynomial library of degree 1 that includes a constant
    # This should produce features: ['1', 'x1', 'v1', 'x2', 'v2', 'u'] if set correctly
    library = ps.PolynomialLibrary(degree=2, include_interaction=True, include_bias=True)
    model = ps.SINDy(feature_library=library, feature_names=["x1","v1","x2","v2","u"])

    # Fit the model (PySINDy fits all state derivatives)
    model.fit(X, t=t, x_dot=X_dot, u=U)
    coeff_matrix = model.coefficients()  # shape: (n_features, n_states)
    pysindy_feature_names = model.get_feature_names()  # may contain duplicates if library changes

    print("Feature names:", model.get_feature_names())
    print("Coefficient matrix shape:", model.coefficients().shape)

    # Deduplicate features
    unique_features = []
    unique_indices = []
    for i, f in enumerate(pysindy_feature_names):
        if f not in unique_features:
            unique_features.append(f)
            unique_indices.append(i)

    
    coeff_matrix = np.array(coeff_matrix).T
    
    # Restrict coeff_matrix to unique features
    coeff_matrix = coeff_matrix[unique_indices, :]
    pysindy_feature_names = unique_features

    # We know our state order is [x1, v1, x2, v2] and hence derivatives [x1_dot, v1_dot, x2_dot, v2_dot].
    # v1_dot is at index 1, v2_dot is at index 3
    coeffs_v1_all = coeff_matrix[:,1]
    coeffs_v2_all = coeff_matrix[:,3]

    # Extract the constant terms (theta_0_1 and theta_0_2)
    if '1' in pysindy_feature_names:
        const_idx = pysindy_feature_names.index('1')
        theta_0_1 = coeffs_v1_all[const_idx]
        theta_0_2 = coeffs_v2_all[const_idx]
    else:
        # If no constant found, assume zero
        theta_0_1 = 0.0
        theta_0_2 = 0.0
        const_idx = None

    # Remove the constant from the feature set and coefficients
    mask = np.ones_like(coeffs_v1_all, dtype=bool)
    if const_idx is not None:
        mask[const_idx] = False

    coeffs_v1_filtered = coeffs_v1_all[mask]
    coeffs_v2_filtered = coeffs_v2_all[mask]
    filtered_features = [f for i,f in enumerate(pysindy_feature_names) if mask[i]]

    # The final desired set of features (no constant) for each equation is:
    final_feature_list = ['x1','v1','x2','v2','u']

    # Initialize arrays to hold final coefficients in correct order
    final_coeffs_v1 = np.zeros(len(final_feature_list))
    final_coeffs_v2 = np.zeros(len(final_feature_list))

    # Map PySINDy coefficients to the final feature order
    for i, feat in enumerate(final_feature_list):
        # Find all occurrences of feat in filtered_features
        feat_indices = [j for j,ff in enumerate(filtered_features) if ff == feat]
        if len(feat_indices) == 0:
            # If the feature was not identified by PySINDy, assume zero coefficient
            final_coeffs_v1[i] = 0.0
            final_coeffs_v2[i] = 0.0
        else:
            # If duplicates exist, average them
            final_coeffs_v1[i] = np.mean(coeffs_v1_filtered[feat_indices])
            final_coeffs_v2[i] = np.mean(coeffs_v2_filtered[feat_indices])

    # Assign some reasonable initial guesses for sigma terms
    sigma_epsilon_1 = 0.1
    sigma_epsilon_2 = 0.1

    # Construct the full parameter vector
    # [theta_0_1, c_v1_x1, c_v1_v1, c_v1_x2, c_v1_v2, c_v1_u,
    #  theta_0_2, c_v2_x1, c_v2_v1, c_v2_x2, c_v2_v2, c_v2_u,
    #  sigma_epsilon_1, sigma_epsilon_2]
    initial_guess = [theta_0_1] + list(final_coeffs_v1) + [theta_0_2] + list(final_coeffs_v2) + [sigma_epsilon_1, sigma_epsilon_2]

    return np.array(initial_guess)
