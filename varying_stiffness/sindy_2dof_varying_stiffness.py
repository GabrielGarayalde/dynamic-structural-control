# sindy_2dof_varying_stiffness.py

import numpy as np
import pysindy as ps
import pandas as pd

###############################################################################
# BUILD & FIT THE SINDy MODEL
###############################################################################
def build_sindy_model_for_varying_stiffness(
    poly_degree=2,
    include_bias=True,
    include_interactions=True,
    stlsq_threshold=0.01,
    stlsq_alpha=0.0,
    stlsq_max_iter=1e5
):
    """
    Build a PySINDy model that includes alpha(t) as a control input.

    We use a PolynomialLibrary of specified degree with optional interactions & bias,
    and an STLSQ optimizer with user-defined threshold and alpha (ridge).
    """
    library = ps.PolynomialLibrary(
        degree=poly_degree,
        include_interaction=include_interactions,
        include_bias=include_bias
    )

    optimizer = ps.STLSQ(
        threshold=stlsq_threshold,
        alpha=stlsq_alpha,
        max_iter=int(stlsq_max_iter)
    )

    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer
    )
    return model


def fit_sindy_model(
    X, X_dot,
    alpha_array,
    t_array,
    poly_degree=2,
    include_bias=True,
    include_interactions=True,
    stlsq_threshold=0.01,
    stlsq_alpha=0.0,
    stlsq_max_iter=1e5
):
    """
    Fit the SINDy model to the data (X, X_dot, alpha).
    Returns the fitted model, the coefficient matrix, and feature names.
    """
    model = build_sindy_model_for_varying_stiffness(
        poly_degree=poly_degree,
        include_bias=include_bias,
        include_interactions=include_interactions,
        stlsq_threshold=stlsq_threshold,
        stlsq_alpha=stlsq_alpha,
        stlsq_max_iter=stlsq_max_iter
    )

    # alpha_array must be 2D for pysindy (N, n_controls=1)
    alpha_2d = alpha_array.reshape(-1, 1)

    model.fit(
        X,
        t=t_array[1] - t_array[0],  # or pass t_array directly for more advanced usage
        x_dot=X_dot,
        u=alpha_2d
    )

    coeffs = model.coefficients()  # shape: (n_states, n_features)
    feature_names = model.get_feature_names()  # e.g. ['x0', 'x1', 'u0', 'x0^2', ...]

    return model, coeffs, feature_names


###############################################################################
# COMPARISON & PRUNING (OPTIONAL, SIMILAR TO BEFORE)
###############################################################################
def prune_sindy_features(model, rows_for_coeffs=(1,3), tol=1e-6):
    """
    Identify features with absolute coefficient > tol in any of the specified rows,
    then return a pruned coefficient matrix and a pruned list of feature names.

    By default, rows_for_coeffs=(1,3) => v1_dot is state index=1, v2_dot=3,
    if your states are [x1, v1, x2, v2].
    """
    coeff_matrix = model.coefficients()
    # Extract only the relevant rows
    row1 = coeff_matrix[rows_for_coeffs[0], :]
    row2 = coeff_matrix[rows_for_coeffs[1], :]

    # Identify columns (features) that are above tol
    idx1 = np.where(np.abs(row1) > tol)[0]
    idx2 = np.where(np.abs(row2) > tol)[0]
    active_feats = np.union1d(idx1, idx2)

    # Build pruned matrix
    pruned_row1 = row1[active_feats]
    pruned_row2 = row2[active_feats]
    pruned_coeff_matrix = np.vstack([pruned_row1, pruned_row2])

    # Pruned feature names
    all_feat_names = model.get_feature_names()
    pruned_feat_names = [all_feat_names[i] for i in active_feats]

    return pruned_coeff_matrix, pruned_feat_names, active_feats


def compare_coeffs(true_coeffs, estimated_coeffs, feature_names):
    """
    Simple DataFrame comparison of two arrays of coefficients, same length,
    along with provided feature names. Returns a DataFrame.
    """
    arr_true = np.array(true_coeffs).flatten()
    arr_est = np.array(estimated_coeffs).flatten()

    if len(arr_true) != len(arr_est):
        raise ValueError("Mismatch in length of true vs. estimated coefficients.")
    if len(arr_true) != len(feature_names):
        raise ValueError("Mismatch in length of coefficients vs. feature_names.")

    abs_diff = np.abs(arr_true - arr_est)
    pct_diff = [
        100.0 * d / np.abs(t) if np.abs(t) > 1e-14 else np.nan
        for d, t in zip(abs_diff, arr_true)
    ]

    df = pd.DataFrame({
        'Feature': feature_names,
        'True Coeff.': arr_true,
        'Estimated': arr_est,
        'AbsDiff': abs_diff,
        '%Diff': pct_diff
    })

    # Round numeric columns
    for col in ['True Coeff.', 'Estimated', 'AbsDiff', '%Diff']:
        df[col] = df[col].round(5)

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



def predict_sindy_trajectory(fitted_model, x0, t, U):
    """
    Predict the trajectory from a SINDy model using simple Euler integration,
    with a check to prevent crashes if predictions explode.

    Parameters
    ----------
    fitted_model : pysindy.SINDy
        The discovered model with .predict(...) available.
    x0 : ndarray, shape (4,)
        Initial condition [x1, v1, x2, v2].
    t : ndarray
        Time array of length N.
    U : ndarray
        1D control input array of length N (e.g. alpha(t)).

    Returns
    -------
    X_pred : ndarray, shape (N, 4)
        The predicted trajectory from the SINDy model under Euler integration.
        If predictions explode, the remaining values in X_pred remain zero.
    """
    X_pred = np.zeros((len(t), len(x0)))
    X_pred[0] = x0

    dt = t[1] - t[0]
    for i in range(len(t) - 1):
        try:
            # Current state
            x_current = X_pred[i].reshape(1, -1)  # shape (1,4)
            # Current control => reshape to (1,1)
            u_current = np.array([[U[i]]])        # or U[i].reshape(1,1)
            
            # Predict derivative
            x_dot_pred = fitted_model.predict(x_current, u=u_current)[0]  # shape (4,)
            
            # Euler step
            X_pred[i + 1] = X_pred[i] + dt * x_dot_pred

            # Check for invalid or exploding values
            if not np.isfinite(X_pred[i + 1]).all():
                print(f"Warning: Prediction exploded at time step {i + 1} (t = {t[i + 1]}).")
                break

        except Exception as e:
            print(f"Error at time step {i + 1} (t = {t[i + 1]}): {e}")
            break

    return X_pred


