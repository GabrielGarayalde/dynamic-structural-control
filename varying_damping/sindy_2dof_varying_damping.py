import numpy as np
import pysindy as ps
import pandas as pd

###############################################################################
# BUILD & FIT THE SINDy MODEL
###############################################################################
def build_sindy_model_for_varying_damping(
    poly_degree=2,
    include_bias=True,
    include_interactions=True,
    stlsq_threshold=0.01,
    stlsq_alpha=0.0,
    stlsq_max_iter=1e5
):
    """
    Build a PySINDy model that includes alpha(t) as a control input,
    for the case where alpha(t) splits the total damping.
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
    Fit the SINDy model to the data (X, X_dot) with alpha(t) in [0,1].
    The discovered model is for a 2DOF system with varying damping ratio.
    """
    model = build_sindy_model_for_varying_damping(
        poly_degree=poly_degree,
        include_bias=include_bias,
        include_interactions=include_interactions,
        stlsq_threshold=stlsq_threshold,
        stlsq_alpha=stlsq_alpha,
        stlsq_max_iter=stlsq_max_iter
    )

    alpha_2d = alpha_array.reshape(-1, 1)  # shape (N,1)

    model.fit(
        X,
        t=t_array[1] - t_array[0],
        x_dot=X_dot,
        u=alpha_2d
    )

    coeffs = model.coefficients()  # shape: (n_states, n_features)
    feature_names = model.get_feature_names()
    return model, coeffs, feature_names


###############################################################################
# FEATURE PRUNING & COMPARISON
###############################################################################
def prune_sindy_features(model, rows_for_coeffs=(1,3), tol=1e-6):
    """
    Identify features with absolute coefficient > tol in v1_dot (row=1)
    or v2_dot (row=3), return a pruned matrix & pruned list of features.
    """
    coeff_matrix = model.coefficients()
    row_v1dot = coeff_matrix[rows_for_coeffs[0], :]
    row_v2dot = coeff_matrix[rows_for_coeffs[1], :]

    idx_v1 = np.where(np.abs(row_v1dot) > tol)[0]
    idx_v2 = np.where(np.abs(row_v2dot) > tol)[0]
    active_features = np.union1d(idx_v1, idx_v2)

    pruned_coeff_matrix = np.vstack([
        row_v1dot[active_features],
        row_v2dot[active_features]
    ])
    feat_names_all = model.get_feature_names()
    pruned_feat_names = [feat_names_all[i] for i in active_features]

    return pruned_coeff_matrix, pruned_feat_names, active_features


def compare_coeffs(true_coeffs, estimated_coeffs, feature_names):
    """
    Simple DataFrame comparison of two arrays of coefficients.
    """
    arr_true = np.array(true_coeffs).flatten()
    arr_est = np.array(estimated_coeffs).flatten()
    if len(arr_true) != len(arr_est):
        raise ValueError("Mismatch in length of true vs. estimated coefficients.")
    if len(arr_true) != len(feature_names):
        raise ValueError("Mismatch in length of coefficients vs. feature_names.")

    abs_diff = np.abs(arr_true - arr_est)
    pct_diff = []
    for (d, t) in zip(abs_diff, arr_true):
        if np.abs(t) > 1e-14:
            pct_diff.append(100.0 * d / np.abs(t))
        else:
            pct_diff.append(np.nan)

    df = pd.DataFrame({
        'Feature': feature_names,
        'True Coeff.': arr_true.round(5),
        'Estimated': arr_est.round(5),
        'AbsDiff': abs_diff.round(5),
        '%Diff': np.round(pct_diff, 5)
    })
    return df


def predict_sindy_trajectory(fitted_model, x0, t, U):
    """
    Predict trajectory from a SINDy model with single control U(t).
    Simple Euler integration as an example.
    """
    X_pred = np.zeros((len(t), len(x0)))
    X_pred[0] = x0
    dt = t[1] - t[0]

    for i in range(len(t) - 1):
        x_current = X_pred[i].reshape(1, -1)
        u_current = np.array([[U[i]]])  # shape (1,1)
        x_dot = fitted_model.predict(x_current, u=u_current)[0]
        X_pred[i+1] = X_pred[i] + dt * x_dot
    return X_pred
