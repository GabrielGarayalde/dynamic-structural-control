import numpy as np
import pysindy as ps
import pandas as pd

def build_sindy_model_for_varying_damping(
    poly_degree=2,
    include_bias=True,
    include_interactions=True,
    stlsq_threshold=0.01,
    stlsq_alpha=0.0,
    stlsq_max_iter=1e5
):
    """
    Build a PySINDy model that can accept 2 input channels:
      u0 = alpha(t)  (the single control)
      u1 = sin(forcing_freq * t) (the known environment forcing)
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
    sin_forcing_array,
    t_array,
    poly_degree=2,
    include_bias=True,
    include_interactions=True,
    stlsq_threshold=0.01,
    stlsq_alpha=0.0,
    stlsq_max_iter=1e5
):
    """
    Fit the SINDy model for a 2DOF system with:
      - alpha(t) => control input #1
      - sin_forcing_array => environment forcing input #2
    """
    model = build_sindy_model_for_varying_damping(
        poly_degree=poly_degree,
        include_bias=include_bias,
        include_interactions=include_interactions,
        stlsq_threshold=stlsq_threshold,
        stlsq_alpha=stlsq_alpha,
        stlsq_max_iter=stlsq_max_iter
    )

    U_2d = np.column_stack([alpha_array, sin_forcing_array])  # shape (N,2)

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
    Identify features with abs(coeff) > tol in v1_dot (row=1) or v2_dot (row=3).
    """
    coeff_matrix = model.coefficients()
    row_v1 = coeff_matrix[rows_for_coeffs[0], :]
    row_v2 = coeff_matrix[rows_for_coeffs[1], :]

    idx1 = np.where(np.abs(row_v1) > tol)[0]
    idx2 = np.where(np.abs(row_v2) > tol)[0]
    active = np.union1d(idx1, idx2)

    pruned_v1 = row_v1[active]
    pruned_v2 = row_v2[active]
    pruned_coeff_matrix = np.vstack([pruned_v1, pruned_v2])

    all_feat_names = model.get_feature_names()
    pruned_feat_names = [all_feat_names[i] for i in active]

    return pruned_coeff_matrix, pruned_feat_names, active


def compare_coeffs(true_coeffs, estimated_coeffs, feature_names):
    """
    Compare an array of true coefficients vs. discovered in a DataFrame.
    """
    import pandas as pd
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
            pct_diff.append(100.0 * abs(t_val - e_val)/abs(t_val))
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
    Simple Euler integration with 2 inputs => [alpha(t), sin_forcing(t)].
    """
    X_pred = np.zeros((len(t), len(x0)))
    X_pred[0] = x0
    dt = t[1] - t[0]

    for i in range(len(t)-1):
        x_now = X_pred[i].reshape(1, -1)
        u_now = np.array([[alpha_array[i], sin_forcing_array[i]]])
        x_dot = fitted_model.predict(x_now, u=u_now)[0]
        X_pred[i+1] = X_pred[i] + dt*x_dot

    return X_pred
