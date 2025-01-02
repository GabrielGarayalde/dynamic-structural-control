# sindy_2dof.py

import numpy as np
import pandas as pd
import pysindy as ps

###############################################################################
# A) CENTRAL CONFIG FOR SINDy LIBRARY
###############################################################################
POLY_DEGREE = 2
INCLUDE_BIAS = True
INCLUDE_INTERACTIONS = False  # or True, if you want cross-terms

def build_sindy_model(
    poly_degree=POLY_DEGREE,
    include_bias=INCLUDE_BIAS,
    include_interactions=INCLUDE_INTERACTIONS
):
    """
    Build and return a PySINDy model that uses a PolynomialLibrary
    with the specified degree, bias, and interactions.
    """
    library = ps.PolynomialLibrary(
        degree=poly_degree,
        include_interaction=include_interactions,
        include_bias=include_bias
    )
    model = ps.SINDy(feature_library=library)
    return model


def get_initial_guess_from_pysindy(
    X, X_dot, U=None, t=None,
    rows_for_coeffs=(1, 3),
    sigma_epsilon_1=0.1,
    sigma_epsilon_2=0.1,
    poly_degree=POLY_DEGREE,
    include_bias=INCLUDE_BIAS,
    include_interactions=INCLUDE_INTERACTIONS
):
    """
    Fit a PySINDy model to (X, X_dot, U) and extract an initial guess vector:
      [coeffs_row1..., coeffs_row2..., sigma_epsilon_1, sigma_epsilon_2].

    Returns
    -------
    initial_guess : ndarray
        Flattened array of size (2 * n_features + 2) if 2 rows_for_coeffs.
    feature_names : list of str
        SINDy feature names, e.g. ['1','x0','x1','u0','x0^2','x0*x1'].
    model : pysindy.SINDy
        The fitted PySINDy model.
    """

    # 1) Build & fit the model
    model = build_sindy_model(
        poly_degree=poly_degree,
        include_bias=include_bias,
        include_interactions=include_interactions
    )
    model.fit(X, t=t, x_dot=X_dot, u=U)

    # 2) Extract coefficients and feature names
    coeff_matrix = model.coefficients()   # shape (n_states, n_library_features)
    feature_names = model.get_feature_names()

    # 3) Collect desired rows for v1_dot (row=1) and v2_dot (row=3)
    selected_rows = []
    for r in rows_for_coeffs:
        if r < 0 or r >= coeff_matrix.shape[0]:
            raise ValueError(f"Row {r} invalid for coefficient matrix with shape {coeff_matrix.shape}.")
        selected_rows.append(coeff_matrix[r, :])  # shape (n_features,)

    # 4) Flatten them + noise placeholders
    initial_guess = np.concatenate(selected_rows + [[sigma_epsilon_1, sigma_epsilon_2]])

    return initial_guess, feature_names, model


def get_deterministic_params_from_sindy(fitted_sindy_model, rows_for_coeffs=(1,3)):
    """
    Extract the deterministic parameters for v1_dot and v2_dot from the fitted SINDy model.
    Flatten them into one array: [row_v1_dot..., row_v2_dot...]
    """
    coeff_matrix = fitted_sindy_model.coefficients()  # shape (n_states, n_features)
    row_v1_dot = coeff_matrix[rows_for_coeffs[0], :]
    row_v2_dot = coeff_matrix[rows_for_coeffs[1], :]
    theta_det = np.concatenate([row_v1_dot, row_v2_dot])
    return theta_det


def build_expanded_feature_names(base_features, eq_labels=("v1_dot", "v2_dot")):
    """
    Create a final feature name list of length 2*len(base_features)+2, e.g.:
    [v1_dot_f1, v1_dot_f2, ..., v2_dot_f1, v2_dot_f2, ..., sigma_epsilon_1, sigma_epsilon_2].
    """
    expanded_list = []
    # v1_dot features
    for f in base_features:
        expanded_list.append(f"{eq_labels[0]}_{f}")
    # v2_dot features
    for f in base_features:
        expanded_list.append(f"{eq_labels[1]}_{f}")
    # noise
    expanded_list.append("sigma_epsilon_1")
    expanded_list.append("sigma_epsilon_2")
    return expanded_list


def compare_coeffs(true_coeffs, initial_guess, feature_names):
    """
    Build a DataFrame comparing true vs. estimated coefficients element-wise.
    """
    arr_true = np.array(true_coeffs).flatten()
    arr_guess = np.array(initial_guess).flatten()

    # Expand feature_names if needed for the noise terms
    nf = len(feature_names)
    if len(arr_guess) > nf:
        extra = len(arr_guess) - nf
        noise_names = [f"sigma_epsilon_{i+1}" for i in range(extra)]
        feature_names_extended = feature_names + noise_names
    else:
        feature_names_extended = feature_names

    if len(arr_true) != len(arr_guess):
        raise ValueError(f"Lengths differ: true={len(arr_true)} vs guess={len(arr_guess)}")
    if len(feature_names_extended) != len(arr_guess):
        raise ValueError(f"Feature names length mismatch.")

    abs_diff = np.abs(arr_true - arr_guess)
    pct_diff = []
    for tval, diff in zip(arr_true, abs_diff):
        if np.isclose(tval, 0.0):
            pct_diff.append(np.nan)
        else:
            pct_diff.append(100.0 * diff / np.abs(tval))

    df = pd.DataFrame({
        'Feature': feature_names_extended,
        'True Coeff.': arr_true,
        'Initial Guess': arr_guess,
        'Abs. Diff': abs_diff,
        '% Diff': pct_diff
    })
    return df
