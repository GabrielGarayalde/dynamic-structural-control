import numpy as np
from scipy.optimize import minimize
import time


def build_pruned_feature_function(pruned_feature_names):
    lambdas = []
    for feat_str in pruned_feature_names:
        # 1) Trim and handle cross terms
        expr_str = feat_str.strip()
        
        # If your features come in like "x0 x2", you can do:
        parts = expr_str.split()
        
        # If there's more than one part, put '*' between them:
        # => "x0 x2" becomes "x0*x2"
        expr_str = '*'.join(parts)
        
        # 2) Convert '^' -> '**'
        expr_str = expr_str.replace('^', '**')
        
        # 3) Replace x0->x[0], x1->x[1], etc.
        expr_str = expr_str.replace('x0', 'x[0]')
        expr_str = expr_str.replace('x1', 'x[1]')
        expr_str = expr_str.replace('x2', 'x[2]')
        expr_str = expr_str.replace('x3', 'x[3]')
        expr_str = expr_str.replace('u0', 'u_val[0]')
        expr_str = expr_str.replace('u1', 'u_val[1]')

        # 4) If "1", just 1.0
        if expr_str == '1':
            code = "1.0"
        else:
            code = expr_str

        # Build the python lambda
        lam = eval(f"lambda x, u_val: {code}")
        lambdas.append(lam)

    def pruned_feature_func(x, u_val):
        return np.array([f(x, u_val) for f in lambdas], dtype=float)

    return pruned_feature_func



def build_2dof_step_pruned(pruned_feature_func, pruned_coeff_matrix):
    """
    pruned_feature_func(x, [u1, u2]) => (n_features,)
    pruned_coeff_matrix => shape (2, n_features) [row0 => v1_dot, row1 => v2_dot]
    """
    def identified_model_step(x, u_vec, dt):
        # u_vec = [u1, u2]
        feats = pruned_feature_func(x, u_vec)
        v1_dot = np.dot(pruned_coeff_matrix[0], feats)
        v2_dot = np.dot(pruned_coeff_matrix[1], feats)

        x1, v1, x2, v2 = x
        x1_next = x1 + dt*v1
        v1_next = v1 + dt*v1_dot
        x2_next = x2 + dt*v2
        v2_next = v2 + dt*v2_dot
        return np.array([x1_next, v1_next, x2_next, v2_next])

    return identified_model_step


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
    Perform an MPC routine using the custom step function for a 2DOF system,
    but now each time step has a 2D control [u1, u2].
    We'll flatten the horizon control into length 2*N inside the optimizer.
    """
    dt = t[1] - t[0]
    n_mpc_samples = len(sigma_draws)
    X_all = np.zeros((n_mpc_samples, len(t), 4))
    U_all = np.zeros((n_mpc_samples, len(t)-1, 2))  # store 2D control at each step

    # Build pruned feature func
    pruned_feature_func = build_pruned_feature_function(pruned_feature_names)
    # Build step function
    identified_model_step = build_2dof_step_pruned(pruned_feature_func, pruned_coeff_matrix)

    def mpc_cost(u_sequence, current_state, horizon):
        """
        u_sequence: shape (2*horizon,) flattened
        """
        u_seq_2d = u_sequence.reshape((horizon, 2))
        x_pred = current_state.copy()
        cost = 0.0
        for k in range(horizon):
            err = x_pred - x_ref
            # If R is scalar, we do cost += R*(u1^2 + u2^2),
            # or if R is 2x2, do cost += [u1,u2]^T R [u1,u2].
            if np.isscalar(R):
                cost += err.T @ Q @ err + R*(u_seq_2d[k,0]**2 + u_seq_2d[k,1]**2)
            else:
                cost += err.T @ Q @ err + u_seq_2d[k] @ R @ u_seq_2d[k]
            x_pred = identified_model_step(x_pred, u_seq_2d[k], dt)
        return cost

    for s in range(n_mpc_samples):
        sigma1, sigma2 = sigma_draws[s]
        print(f"\nMPC sample {s}: sigma1={sigma1:.3f}, sigma2={sigma2:.3f}")

        x_current = x0.copy()
        X_sim = [x_current]
        U_sim = []

        for idx in range(len(t)-1):
            if idx % 100 == 0:
                print(idx)
            remain = len(t) - idx - 1
            horizon = min(N, remain)

            def cost_function(u_seq_flat):
                return mpc_cost(u_seq_flat, x_current, horizon)

            # Initial guess for horizon controls: zeros
            u_init = np.zeros(2*horizon)
            # Bounds => each of the 2*horizon entries in [u_min, u_max]
            bounds = [(u_min, u_max)] * (2*horizon)

            res = minimize(cost_function, u_init, method='SLSQP',
                           bounds=bounds, options={'maxiter':50, 'ftol':1e-4})
            if not res.success:
                print(f"  [WARNING] SLSQP fail at t index {idx}, reason: {res.message}")
                # fallback
                u_opt_flat = np.zeros(2*horizon)
            else:
                u_opt_flat = res.x

            # Extract the first control from the solution
            u_opt_2d = u_opt_flat.reshape((horizon, 2))
            chosen_u = u_opt_2d[0]  # (u1, u2) at this step

            # Step the system with noise
            x_next = identified_model_step(x_current, chosen_u, dt)
            # Add random noise to v1, v2
            noise1 = np.random.normal(0, sigma1)
            noise2 = np.random.normal(0, sigma2)
            x_next[1] += dt * noise1  # v1
            x_next[3] += dt * noise2  # v2

            U_sim.append(chosen_u)
            X_sim.append(x_next)
            x_current = x_next

        X_sim = np.array(X_sim)
        U_sim = np.array(U_sim)
        # For simplicity, fill any leftover from horizon edges
        # so we can store them in X_all, U_all
        if len(U_sim) >= 5:
            U_sim[-4:] = U_sim[-5]
        X_all[s, :len(X_sim), :] = X_sim
        U_all[s, :len(U_sim), :] = U_sim

    # Summaries
    X_mean = np.mean(X_all, axis=0)
    X_std  = np.std(X_all, axis=0)
    U_mean = np.mean(U_all, axis=0)
    U_std  = np.std(U_all, axis=0)

    return X_all, U_all, X_mean, X_std, U_mean, U_std
