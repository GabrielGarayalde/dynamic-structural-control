"2DOF Bayesian MPC"

import numpy as np
from scipy.optimize import minimize
import time


def build_pruned_feature_function(pruned_feature_names):
    """
    Build a custom function that computes only the pruned features
    (the ones that have nonzero coefficients).
    """
    lambdas = []
    for feat_str in pruned_feature_names:
        # 1) Trim and handle cross terms
        expr_str = feat_str.strip()
        parts = expr_str.split()
        expr_str = '*'.join(parts)

        # 2) Convert '^' -> '**'
        expr_str = expr_str.replace('^', '**')

        # 3) Replace x0->x[0], etc.
        expr_str = expr_str.replace('x0', 'x[0]')
        expr_str = expr_str.replace('x1', 'x[1]')
        expr_str = expr_str.replace('x2', 'x[2]')
        expr_str = expr_str.replace('x3', 'x[3]')
        expr_str = expr_str.replace('u0', 'u_val')

        # If "1", just 1.0
        if expr_str == '1':
            code = "1.0"
        else:
            code = expr_str

        lam = eval(f"lambda x, u_val: {code}")
        lambdas.append(lam)

    def pruned_feature_func(x, u_val):
        return np.array([f(x, u_val) for f in lambdas], dtype=float)

    return pruned_feature_func


def build_2dof_step_pruned(pruned_feature_func, pruned_coeff_matrix):
    """
    pruned_feature_func : function that returns (n_pruned_features,) from (x, u_val)
    pruned_coeff_matrix : shape (2, n_pruned_features)
                          row 0 => v1_dot
                          row 1 => v2_dot
    """
    def identified_model_step(x, u_val, dt):
        feats = pruned_feature_func(x, u_val)
        v1_dot = np.dot(pruned_coeff_matrix[0], feats)
        v2_dot = np.dot(pruned_coeff_matrix[1], feats)

        x1, v1, x2, v2 = x
        x1_next = x1 + dt*v1
        v1_next = v1 + dt*v1_dot
        x2_next = x2 + dt*v2
        v2_next = v2 + dt*v2_dot
        return np.array([x1_next, v1_next, x2_next, v2_next])

    return identified_model_step


# ==============================================================
# 4) RUN A BAYESIAN MPC WITH THE CUSTOM STEP FUNCTION
# ==============================================================
def run_bayesian_mpc_deterministic_noise(
    pruned_feature_names,
    pruned_coeff_matrix,
    model,
    sigma_draws,
    x0, t, Q, R, N,
    u_min, u_max, x_ref,
    rows_for_coeffs=(1,3)
):
    """
    Perform an MPC routine using the custom step function for a 2DOF system:
      v1_dot = ...
      v2_dot = ...

    model : fitted PySINDy model
    sigma_draws : array of shape (n_samples,2) => noise stdevs [sigma1, sigma2]
    x0 : initial state (4,) => [x1, v1, x2, v2]
    t : time array
    Q,R : cost weighting
    N : horizon steps
    u_min, u_max : control bounds
    x_ref : target state
    rows_for_coeffs : typically (1,3) for v1_dot, v2_dot

    Returns
    -------
    X_all, U_all : shape (n_samples, len(t), 4) and (n_samples, len(t)-1)
    X_mean, X_std, U_mean, U_std
    """
    
    # Start timing
    start_time = time.time()

    dt = t[1] - t[0]
    n_mpc_samples = len(sigma_draws)
    X_all = np.zeros((n_mpc_samples, len(t), 4))
    U_all = np.zeros((n_mpc_samples, len(t)-1))


    # 5) Build pruned feature func
    pruned_feature_func = build_pruned_feature_function(pruned_feature_names)
    
    # 6) Build step function
    identified_model_step = build_2dof_step_pruned(pruned_feature_func, pruned_coeff_matrix)


    # Cost function for a finite horizon
    def mpc_cost(u_sequence, current_state, horizon):
        x_pred = current_state.copy()
        cost = 0.0

        for u_k in u_sequence[:horizon]:
            # State penalty
            error = x_pred - x_ref
            cost += error.T @ Q @ error
            # Control penalty
            cost += R*(u_k**2)
            # Nominal forward step (no noise)
            x_pred = identified_model_step(x_pred, u_k, dt)

        return cost

    # =========================
    # MPC Loop over each sample
    # =========================

    for s in range(n_mpc_samples):
        sigma1, sigma2 = sigma_draws[s]
        print(f"\nMPC sample {s}: sigma1={sigma1:.3f}, sigma2={sigma2:.3f}")

        x_current = x0.copy()
        X_sim = [x_current]
        U_sim = []

        for idx in range(len(t)-1):
            remain = len(t)-idx-1
            horizon = min(N, remain)

            def cost_function(u_seq):
                return mpc_cost(u_seq, x_current, horizon)

            # Solve the horizon input sequence
            u_init = np.zeros(horizon)
            bounds = [(u_min, u_max)]*horizon

            res = minimize(cost_function, u_init, method='SLSQP',
                            bounds=bounds, options={'maxiter':50, 'ftol':1e-4})
            if res.success:
                u_opt = res.x[0]
                # print(idx)
            else:
                print(f"  [WARNING] SLSQP fail at t index {idx}, reason: {res.message}")
                u_opt = 0.0

            # Step the real system with noise
            x_next = identified_model_step(x_current, u_opt, dt)
            # Add random noise in v1, v2
            noise1 = np.random.normal(0, sigma1)
            noise2 = np.random.normal(0, sigma2)
            x_next[1] += dt*noise1  # v1
            x_next[3] += dt*noise2  # v2

            U_sim.append(u_opt)
            X_sim.append(x_next)
            x_current = x_next

        X_sim = np.array(X_sim)
        U_sim = np.array(U_sim)
        # X_all[s, :len(X_sim), :] = X_sim
        # U_all[s, :len(U_sim)] = U_sim
        # Ensure the last 4 entries of U_sim are the same as the 5th last entry
        if len(U_sim) >= 5:
            U_sim[-4:] = U_sim[-5]
        elif len(U_sim) > 0:  # Handle edge case where U_sim < 5
            U_sim[-len(U_sim):] = U_sim[0]
        
        X_all[s, :len(X_sim), :] = X_sim
        U_all[s, :len(U_sim)] = U_sim

    # End timing
    end_time = time.time()
    print(f"\n=== MPC loop completed in {end_time - start_time:.2f} seconds ===")
    
    
    # Summaries
    X_mean = np.mean(X_all, axis=0)
    X_std  = np.std(X_all, axis=0)
    U_mean = np.mean(U_all, axis=0)
    U_std  = np.std(U_all, axis=0)

    return X_all, U_all, X_mean, X_std, U_mean, U_std
