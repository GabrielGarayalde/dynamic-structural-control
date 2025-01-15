# sindy_1dof.py

import numpy as np
import pysindy as ps
import pandas as pd

###############################################################################
# A) CENTRAL CONFIG FOR SINDy LIBRARY
###############################################################################
POLY_DEGREE = 2
INCLUDE_BIAS = True
INCLUDE_INTERACTIONS = False

def build_sindy_model(
    poly_degree=POLY_DEGREE,
    include_bias=INCLUDE_BIAS,
    include_interactions=INCLUDE_INTERACTIONS,
    stlsq_threshold=0.1,
    stlsq_alpha=0.1,
    stlsq_max_iter=10000
):
    """
    Build and return a PySINDy model that uses:
     - PolynomialLibrary of specified degree, bias, interactions
     - STLSQ optimizer with user-defined threshold, alpha, etc.
    """
    library = ps.PolynomialLibrary(
        degree=poly_degree,
        include_interaction=include_interactions,
        include_bias=include_bias
    )

    optimizer = ps.STLSQ(
        threshold=stlsq_threshold,
        alpha=stlsq_alpha,
        max_iter=stlsq_max_iter
    )

    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer
    )
    return model

def get_initial_guess_from_pysindy(
    X, X_dot, U=None, t=None,
    row_for_coeffs=(1,),
    poly_degree=POLY_DEGREE,
    include_bias=INCLUDE_BIAS,
    include_interactions=INCLUDE_INTERACTIONS
):
    """
    Fit a PySINDy model to (X, X_dot, U) and extract an initial guess vector
    for the v_dot equation (row_for_coeffs = 1).

    Returns
    -------
    initial_guess : ndarray
        Flattened array for the single row v_dot.
    feature_names : list of str
        SINDy feature names, e.g. ['1','x0','x1','u0','x0^2'].
    model : pysindy.SINDy
        The fitted PySINDy model.
    """
    model = build_sindy_model(
        poly_degree=poly_degree,
        include_bias=include_bias,
        include_interactions=include_interactions
    )
    model.fit(X, t=t, x_dot=X_dot, u=U)

    coeff_matrix = model.coefficients()   # shape (n_states, n_library_features)
    feature_names = model.get_feature_names()

    # row_for_coeffs=(1,) typically means row=1 => v_dot, because row=0 => x_dot
    row_vdot = row_for_coeffs[0]

    # Extract that row (n_features,)
    selected_row = coeff_matrix[row_vdot, :]

    # Return the flattened row
    initial_guess = selected_row
    return initial_guess, feature_names, model

def build_expanded_feature_names(base_features, eq_label="v_dot"):
    """
    Create a final feature name list of length len(base_features)
    e.g.: [v_dot_f1, v_dot_f2, ..., v_dot_fN].
    """
    expanded_list = [f"{eq_label}_{f}" for f in base_features]
    return expanded_list

def compare_coeffs(true_coeffs, initial_guess, feature_names,
                   active_feature_indices=None):
    """
    Compare true vs. estimated coefficients for a single equation (v_dot).
    """
    arr_true = np.array(true_coeffs).flatten()
    arr_guess = np.array(initial_guess).flatten()

    if len(arr_true) != len(arr_guess):
        raise ValueError(
            f"Lengths differ: true={len(arr_true)} vs guess={len(arr_guess)}"
        )
    if len(feature_names) != len(arr_guess):
        raise ValueError(
            f"Feature names length mismatch: {len(feature_names)} vs {len(arr_guess)}"
        )

    if active_feature_indices is not None:
        arr_true = arr_true[active_feature_indices]
        arr_guess = arr_guess[active_feature_indices]
        feature_names = [feature_names[i] for i in active_feature_indices]

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

    numeric_cols = ['True Coeff.', 'Initial Guess', 'Abs. Diff', '% Diff']
    df[numeric_cols] = df[numeric_cols].round(3)

    return df

def prune_sindy_features(model, row_for_coeffs=(1,), tol=1e-6):
    """
    Identify features with absolute coefficient > tol in the v_dot row,
    then return a pruned coefficient vector and feature names.

    Returns
    -------
    pruned_coeff_vector : ndarray, shape (n_pruned_features,)
    pruned_feature_names : list of str, length n_pruned_features
    active_feature_indices : ndarray of shape (n_pruned_features,)
    """
    coeff_matrix = model.coefficients()  # shape (n_states, n_features)
    row_vdot = coeff_matrix[row_for_coeffs[0], :]

    active_features = np.where(np.abs(row_vdot) > tol)[0]

    pruned_coeff_vector = row_vdot[active_features]
    original_feature_names = model.get_feature_names()
    pruned_feature_names = [original_feature_names[i] for i in active_features]

    return pruned_coeff_vector, pruned_feature_names, active_features


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