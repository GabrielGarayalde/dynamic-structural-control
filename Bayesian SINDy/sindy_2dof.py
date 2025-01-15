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
    include_interactions=INCLUDE_INTERACTIONS,
    stlsq_threshold=0.1,        # <-- NEW ARG: sparsity threshold
    stlsq_alpha=0.1,             # <-- optional ridge penalty
    stlsq_max_iter=10000         # <-- optional iteration limit
):
    """
    Build and return a PySINDy model that uses:
     - PolynomialLibrary of specified degree, bias, interactions
     - STLSQ optimizer with user-defined threshold, alpha, etc.
    """
    # 1) Define the polynomial library
    library = ps.PolynomialLibrary(
        degree=poly_degree,
        include_interaction=include_interactions,
        include_bias=include_bias
    )

    # 2) Define the STLSQ optimizer
    optimizer = ps.STLSQ(
        threshold=stlsq_threshold,
        alpha=stlsq_alpha,
        max_iter=stlsq_max_iter
    )

    # 3) Build the SINDy model with the custom optimizer
    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer
    )
    
    return model


def get_initial_guess_from_pysindy(
    X, X_dot, U=None, t=None,
    rows_for_coeffs=(1, 3),
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
    initial_guess = np.concatenate(selected_rows)

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

    return expanded_list


import numpy as np
import pandas as pd

def compare_coeffs(true_coeffs, initial_guess, feature_names,
                       active_feature_indices=None):
    """
    Compare true vs. estimated coefficients for a 2-equation system
    (e.g., v1_dot, v2_dot), each with M features => total 2*M coefficients.
    We do *not* include sigma_epsilon terms here.

    Parameters
    ----------
    true_coeffs : array-like, shape (2*M,)
        The "true" values for [v1_dot, v2_dot].
    initial_guess : array-like, shape (2*M,)
        The estimated values from PySINDy or pruned approach.
    feature_names : list of str, length (2*M)
        Typically from build_expanded_feature_names_no_noise(...).
    active_feature_indices : array of int, optional
        Indices in [0..M-1] that are active for *one* equation. We'll replicate for eq2
        by adding M. If None, we show all 2*M.

    Returns
    -------
    df : pd.DataFrame
        Columns: ['Feature','True Coeff.','Initial Guess','Abs. Diff','% Diff']
    """
    arr_true = np.array(true_coeffs).flatten()
    arr_guess = np.array(initial_guess).flatten()

    # Basic sanity checks
    if len(arr_true) != len(arr_guess):
        raise ValueError(
            f"Lengths differ: true={len(arr_true)} vs guess={len(arr_guess)}"
        )
    if len(feature_names) != len(arr_guess):
        raise ValueError(
            f"Feature names length mismatch with guess: {len(feature_names)} vs {len(arr_guess)}"
        )

    # If user provided pruned indices (e.g. [0,2,5]) in [0..M-1], replicate for eq2
    if active_feature_indices is not None:
        # M = half the length of the total
        total_len = len(arr_true)
        M = total_len // 2
        # Build the union of eq1, eq2
        expanded_indices = []
        for i in active_feature_indices:
            expanded_indices.append(i)       # eq1
            expanded_indices.append(i + M)   # eq2

        arr_true = arr_true[expanded_indices]
        arr_guess = arr_guess[expanded_indices]
        feature_names = [feature_names[i] for i in expanded_indices]

    # Build dataframe
    abs_diff = np.abs(arr_true - arr_guess)
    pct_diff = []
    for tval, diff in zip(arr_true, abs_diff):
        if np.isclose(tval, 0.0):
            pct_diff.append(np.nan)
        else:
            pct_diff.append(100.0 * diff / np.abs(tval))

    df = pd.DataFrame({
        'Feature': feature_names,
        'True Coeff.': arr_true,
        'Initial Guess': arr_guess,
        'Abs. Diff': abs_diff,
        '% Diff': pct_diff
    })
    
    # Round all numerical columns to 3 decimal places
    numeric_cols = ['True Coeff.', 'Initial Guess', 'Abs. Diff', '% Diff']
    df[numeric_cols] = df[numeric_cols].round(3)
    
    return df


def prune_sindy_features(model, rows_for_coeffs=(1,3), tol=1e-6):
    """
    Identify features with absolute coefficient > tol in either v1_dot or v2_dot row,
    then return a pruned coefficient matrix (2 rows x #active_features) and
    a pruned list of feature names.

    Parameters
    ----------
    model : fitted PySINDy model
    rows_for_coeffs : tuple
        e.g. (1,3) => row 1 => v1_dot, row 3 => v2_dot
    tol : float
        Tolerance for deciding which coefficients are "nonzero"

    Returns
    -------
    pruned_coeff_matrix : ndarray, shape (2, n_pruned_features)
        row 0 => v1_dot, row 1 => v2_dot
    pruned_feature_names : list of str, length n_pruned_features
    active_feature_indices : ndarray of shape (n_pruned_features,)
        The indices of the features (columns) that remain.
    """
    coeff_matrix = model.coefficients()
    # Extract the two rows we care about (v1_dot, v2_dot)
    row_v1dot = coeff_matrix[rows_for_coeffs[0], :]
    row_v2dot = coeff_matrix[rows_for_coeffs[1], :]

    # Identify which columns (features) are > tol in absolute value
    nonzero_idx_v1 = np.where(np.abs(row_v1dot) > tol)[0]
    nonzero_idx_v2 = np.where(np.abs(row_v2dot) > tol)[0]
    # Union => features used in either eqn
    active_features = np.union1d(nonzero_idx_v1, nonzero_idx_v2)

    # Build a 2-row pruned matrix
    # row 0 => v1_dot, row 1 => v2_dot
    pruned_v1dot = row_v1dot[active_features]
    pruned_v2dot = row_v2dot[active_features]
    pruned_coeff_matrix = np.vstack([pruned_v1dot, pruned_v2dot])

    # Get all original feature names
    original_feature_names = model.get_feature_names()
    # Build the pruned name list
    pruned_feature_names = [original_feature_names[i] for i in active_features]

    return pruned_coeff_matrix, pruned_feature_names, active_features


import numpy as np


def predict_sindy_trajectory(fitted_model, x0, t, U):
    """
    Predict the trajectory from a SINDy model using Euler integration,
    with error handling to prevent crashes if predictions explode.

    Parameters
    ----------
    fitted_model : pysindy.SINDy
        The discovered model with .predict(...) available.
    x0 : ndarray, shape (4,)
        Initial condition for [x1, v1, x2, v2].
    t : ndarray
        Time array.
    U : ndarray
        Control input array, same length as t.

    Returns
    -------
    X_pred : ndarray, shape (len(t), len(x0))
        The predicted trajectory from the SINDy model under Euler integration.
        If predictions explode, the remaining values in X_pred will be zeros.
    """
    X_pred = np.zeros((len(t), len(x0)))
    X_pred[0] = x0

    dt = t[1] - t[0]
    for i in range(len(t) - 1):
        try:
            x_current = X_pred[i].reshape(1, -1)
            u_current = np.array([U[i]]).reshape(1, -1)
            x_dot_pred = fitted_model.predict(x_current, u=u_current)[0]  # shape (4,)
            X_pred[i + 1] = X_pred[i] + dt * x_dot_pred

            # Check for invalid values (NaN, inf, or excessively large values)
            if not np.isfinite(X_pred[i + 1]).all():
                print(f"Warning: Prediction exploded at time step {i + 1} (t = {t[i + 1]}).")
                break  # Stop further predictions but keep array length constant

        except Exception as e:
            print(f"Error at time step {i + 1} (t = {t[i + 1]}): {e}")
            break  # Stop further predictions but keep array length constant

    # Ensure remaining values in X_pred remain as zeros (already initialized as zeros)
    return X_pred

