import numpy as np
import pysindy as ps
import pandas as pd

def build_sindy_model_for_varying_stiffness(
    poly_degree=2,
    include_bias=True,
    include_interactions=True,
    stlsq_threshold=0.01,
    stlsq_alpha=0.0,
    stlsq_max_iter=1e5
):
    """
    Build a PySINDy model that can accept 2 input channels:
      u0 = alpha(t)
      u1 = sin(forcing_freq*t)
    as an example of a known environment forcing plus control.
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
    sin_forcing_array,  # the known environment forcing
    t_array,
    poly_degree=2,
    include_bias=True,
    include_interactions=True,
    stlsq_threshold=0.01,
    stlsq_alpha=0.0,
    stlsq_max_iter=1e5
):
    """
    Fit the SINDy model to data (X, X_dot) with 2 "inputs":
      u0 = alpha(t)
      u1 = sin_forcing(t).
    Even though we only want to *control* alpha(t), we let SINDy see the sinusoid
    so it can correctly capture the dynamics.
    """
    model = build_sindy_model_for_varying_stiffness(
        poly_degree=poly_degree,
        include_bias=include_bias,
        include_interactions=include_interactions,
        stlsq_threshold=stlsq_threshold,
        stlsq_alpha=stlsq_alpha,
        stlsq_max_iter=stlsq_max_iter
    )

    # Build a 2D array: shape (N,2)
    U_2d = np.column_stack([alpha_array, sin_forcing_array])

    dt = t_array[1] - t_array[0]
    model.fit(
        X,
        t=dt,
        x_dot=X_dot,
        u=U_2d
    )

    coeffs = model.coefficients()
    feat_names = model.get_feature_names()
    return model, coeffs, feat_names


def prune_sindy_features(model, rows_for_coeffs=(1,3), tol=1e-6):
    """
    Prune small coefficients from v1_dot, v2_dot.
    """
    coeff_matrix = model.coefficients()
    row1 = coeff_matrix[rows_for_coeffs[0], :]
    row2 = coeff_matrix[rows_for_coeffs[1], :]

    idx1 = np.where(np.abs(row1) > tol)[0]
    idx2 = np.where(np.abs(row2) > tol)[0]
    active_cols = np.union1d(idx1, idx2)

    pruned_v1 = row1[active_cols]
    pruned_v2 = row2[active_cols]
    pruned_coeffs = np.vstack([pruned_v1, pruned_v2])

    all_feat_names = model.get_feature_names()
    pruned_feat_names = [all_feat_names[i] for i in active_cols]
    return pruned_coeffs, pruned_feat_names, active_cols


def compare_coeffs(true_coeffs, estimated_coeffs, feature_names):
    """
    Simple DataFrame comparison of true vs. estimated.
    """
    arr_true = np.array(true_coeffs).flatten()
    arr_est = np.array(estimated_coeffs).flatten()

    if len(arr_true) != len(arr_est):
        raise ValueError("Mismatch in length of true vs. estimated.")
    if len(arr_true) != len(feature_names):
        raise ValueError("Mismatch in length of coefficients vs. feature_names.")

    abs_diff = np.abs(arr_true - arr_est)
    pct_diff = []
    for t_val, e_val in zip(arr_true, arr_est):
        if np.abs(t_val) > 1e-14:
            pct_diff.append(100.0 * abs(t_val - e_val) / np.abs(t_val))
        else:
            pct_diff.append(np.nan)

    df = pd.DataFrame({
        'Feature': feature_names,
        'True Coeff.': arr_true.round(6),
        'Estimated': arr_est.round(6),
        'AbsDiff': abs_diff.round(6),
        '%Diff': np.round(pct_diff, 3)
    })
    return df


def predict_sindy_trajectory(fitted_model, x0, t, alpha_array, sin_forcing_array):
    """
    Predict the system trajectory via Euler integration, with 2 known inputs:
      alpha(t) and sin_forcing(t).
    """
    X_pred = np.zeros((len(t), len(x0)))
    X_pred[0] = x0
    dt = t[1] - t[0]

    for i in range(len(t) - 1):
        x_now = X_pred[i].reshape(1, -1)
        u_now = np.array([[ alpha_array[i], sin_forcing_array[i] ]])  # shape (1,2)
        x_dot = fitted_model.predict(x_now, u=u_now)[0]
        X_pred[i+1] = X_pred[i] + dt*x_dot

    return X_pred
