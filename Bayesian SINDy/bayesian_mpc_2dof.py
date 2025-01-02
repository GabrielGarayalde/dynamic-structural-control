"2DOF Bayesian MPC"

import numpy as np
from scipy.optimize import minimize
import time

# ==============================================================
# 2) BUILD A CUSTOM FEATURE FUNCTION FROM model.get_feature_names()
# ==============================================================
def build_custom_feature_function(model):
    """
    Parse model.get_feature_names(), e.g. ['1', 'x0', 'x1', 'x0^2', 'x0*x1', 'u0^2', ...],
    and build small lambda expressions that compute each feature from (x, u_val).
    
    Returns: feature_func(x, u_val) -> np.ndarray of shape (n_features,).
    
    This is done only once, so we avoid the overhead of library.transform() in a loop.
    """
    feature_names = model.get_feature_names()  # e.g. ['1','x0','x1','x0^2','x0*x1','u0^2',...]

    lambdas = []
    for feat_str in feature_names:
        # 1) Trim leading/trailing spaces (just in case)
        expr_str = feat_str.strip()
        
        # 2) Insert '*' for cross terms:
        parts = expr_str.split()  # split on whitespace
        expr_str = '*'.join(parts) # e.g. ["x0", "x1"] => "x0*x1"
        
        # 3) Replace '^' with '**' for exponent notation
        expr_str = expr_str.replace('^', '**')

        # 2) Replace 'x0','x1','x2','x3' => 'x[0]', 'x[1]', ...
        #    Replace 'u0' => 'u_val'
        expr_str = expr_str.replace('x0', 'x[0]')
        expr_str = expr_str.replace('x1', 'x[1]')
        expr_str = expr_str.replace('x2', 'x[2]')
        expr_str = expr_str.replace('x3', 'x[3]')
        expr_str = expr_str.replace('u0', 'u_val')

        # If it's '1', just a constant
        if expr_str == '1':
            code = "1.0"
        else:
            code = expr_str

        # Build a small lambda: e.g. "lambda x, u_val: x[0]*x[1]"
        lam = eval(f"lambda x, u_val: {code}")
        lambdas.append(lam)

    def feature_func(x, u_val):
        # Evaluate each lambda for the current (x,u_val) => list or array
        return np.array([f(x, u_val) for f in lambdas], dtype=float)

    return feature_func


# ==============================================================
# 3) CONSTRUCT A 2DOF STEP FUNCTION USING THE CUSTOM FEATURES
# ==============================================================
def build_2dof_step_from_feature_func(model, rows_for_coeffs=(1,3)):
    """
    Create a function identified_model_step(x, u_val, dt) that:
      - Builds the feature vector from (x,u_val) using a custom feature function
      - Dots with the relevant SINDy coefficient rows for v1_dot, v2_dot
      - Performs an Euler update
    """

    # 1) Get the SINDy coefficient matrix => shape (n_states, n_features)
    coeff_matrix = model.coefficients()
    # Extract the two rows for v1_dot, v2_dot
    coeffs_v1dot = coeff_matrix[rows_for_coeffs[0], :]  # row=1 => v1_dot
    coeffs_v2dot = coeff_matrix[rows_for_coeffs[1], :]  # row=3 => v2_dot

    # 2) Build the custom feature function
    feature_func = build_custom_feature_function(model)

    # 3) Return an Euler-step function
    def identified_model_step(x, u_val, dt):
        """
        x = [x1, v1, x2, v2]
        u_val = control input
        dt = time step
        """
        # Evaluate features: shape (n_features,)
        feats = feature_func(x, u_val)

        # Dot with the appropriate rows
        v1_dot = np.dot(coeffs_v1dot, feats)
        v2_dot = np.dot(coeffs_v2dot, feats)

        # Euler step
        x1, v1, x2, v2 = x
        x1_next = x1 + dt*v1
        v1_next = v1 + dt*v1_dot
        x2_next = x2 + dt*v2
        v2_next = v2 + dt*v2_dot
        return np.array([x1_next, v1_next, x2_next, v2_next])

    return identified_model_step

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

    # Build the 2DOF Euler step function from the model
    identified_model_step = build_2dof_step_from_feature_func(
        model, rows_for_coeffs=rows_for_coeffs
    )
    
    
    # # 1) Get coefficient matrix
    # coeff_matrix = model.coefficients()
    # row_v1dot = coeff_matrix[1, :]
    # row_v2dot = coeff_matrix[3, :]
    
    # # 2) Identify nonzero columns
    # tol = 1e-6
    # nonzero_idx_v1 = np.where(np.abs(row_v1dot) > tol)[0]
    # nonzero_idx_v2 = np.where(np.abs(row_v2dot) > tol)[0]
    # active_features = np.union1d(nonzero_idx_v1, nonzero_idx_v2)
    
    # # 3) Build pruned coefficient matrix
    # pruned_coeff_matrix = np.vstack([
    #     row_v1dot[active_features],
    #     row_v2dot[active_features]
    # ])
    
    # # 4) Build pruned feature names
    # original_feature_names = model.get_feature_names()
    # pruned_feature_names = [original_feature_names[i] for i in active_features]
    
    # # 5) Build pruned feature func
    # pruned_feature_func = build_pruned_feature_function(pruned_feature_names)
    
    # # 6) Build step function
    # identified_model_step = build_2dof_step_pruned(pruned_feature_func, pruned_coeff_matrix)





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
    from functools import partial

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
