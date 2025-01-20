import numpy as np
from scipy.optimize import minimize
import time

def build_pruned_feature_function_single_control(pruned_feature_names):
    """
    Build a function that, given state x and scalar control u, returns
    the vector of pruned features. For single control, we treat u_val = [u].
    """
    lambdas = []
    for feat_str in pruned_feature_names:
        expr_str = feat_str.strip()

        # Insert '*' between any space-separated tokens (e.g., "x0 x2" => "x0*x2")
        parts = expr_str.split()
        expr_str = '*'.join(parts)

        # Replace '^' with '**' for Python exponent
        expr_str = expr_str.replace('^', '**')

        # Replace x0->x[0], x1->x[1], etc.  and u0->u_val[0]
        expr_str = expr_str.replace('x0', 'x[0]')
        expr_str = expr_str.replace('x1', 'x[1]')
        expr_str = expr_str.replace('x2', 'x[2]')
        expr_str = expr_str.replace('x3', 'x[3]')
        expr_str = expr_str.replace('u0', 'u_val[0]')  # single control => u_val[0]

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


def build_2dof_step_pruned_single_control(pruned_feature_func, pruned_coeff_matrix):
    """
    pruned_feature_func(x, [u]) => shape (n_features,).
    pruned_coeff_matrix => shape (2, n_features) [row0 => v1_dot, row1 => v2_dot].
    """
    def identified_model_step(x, alpha, dt):
        # alpha is a scalar, we pass as [alpha] to the feature func
        feats = pruned_feature_func(x, [alpha])
        v1_dot = np.dot(pruned_coeff_matrix[0], feats)
        v2_dot = np.dot(pruned_coeff_matrix[1], feats)

        x1, v1, x2, v2 = x
        x1_next = x1 + dt*v1
        v1_next = v1 + dt*v1_dot
        x2_next = x2 + dt*v2
        v2_next = v2 + dt*v2_dot
        return np.array([x1_next, v1_next, x2_next, v2_next])

    return identified_model_step


def run_bayesian_mpc_single_control(
    pruned_feature_names,
    pruned_coeff_matrix,
    fitted_model,
    sigma_draws,
    x0, t, Q, R, N,
    alpha_min, alpha_max,
    x_ref,
    rows_for_coeffs=(1,3)
):
    """
    Perform an MPC routine with a single control alpha(t), in [alpha_min, alpha_max].
    We do a horizon of length N. Flatten the horizon control into shape (N,).
    """
    dt = t[1] - t[0]
    n_mpc_samples = len(sigma_draws)
    X_all = np.zeros((n_mpc_samples, len(t), 4))
    U_all = np.zeros((n_mpc_samples, len(t)-1))  # single control at each step

    # Build pruned feature function
    pruned_feature_func = build_pruned_feature_function_single_control(pruned_feature_names)
    # Build step function
    identified_model_step = build_2dof_step_pruned_single_control(pruned_feature_func, pruned_coeff_matrix)

    def mpc_cost(alpha_sequence, current_state, horizon):
        """
        alpha_sequence: shape (horizon,) for the next horizon steps
        """
        x_pred = current_state.copy()
        cost = 0.0
        alpha_base = 0.5  # or whatever baseline you call "uncontrolled"
        S = 0.1  # You choose how large to make the delta penalty

        for k in range(horizon):
            err = x_pred - x_ref
            # If R is scalar, cost += err^T Q err + R*(alpha^2).
            # cost += err.T @ Q @ err + R*(alpha_sequence[k]**2)
            
            # NEW line: penalize distance from alpha_base
            cost += err.T @ Q @ err + R*((alpha_sequence[k] - alpha_base)**2)
            
            # Add a cost on the change in alpha if k>0
            # if k == 0:
            #     # If you want to penalize sudden jump from the *current* alpha
            #     # you could do e.g. (alpha_sequence[0] - alpha_current)**2
            #     delta_alpha = alpha_sequence[k] - alpha_base
            # else:
            #     delta_alpha = alpha_sequence[k] - alpha_sequence[k-1]
            # cost += S*(delta_alpha**2)
            
            x_pred = identified_model_step(x_pred, alpha_sequence[k], dt)
        return cost

    for s in range(n_mpc_samples):
        sigma1, sigma2 = sigma_draws[s]
        print(f"\nMPC sample {s}: sigma1={sigma1:.3f}, sigma2={sigma2:.3f}")

        x_current = x0.copy()
        X_sim = [x_current]
        U_sim = []

        for idx in range(len(t) - 1):
            if idx % 50 == 0:
                print(f"Time index {idx}/{len(t)-1}")

            remain = len(t) - idx - 1
            horizon = min(N, remain)

            def cost_function(alpha_seq):
                return mpc_cost(alpha_seq, x_current, horizon)

            # Initial guess: zeros or middle of [alpha_min, alpha_max]
            alpha_init = np.zeros(horizon)
            
            # Option 1: Start from alpha_base
            alpha_init = np.full(horizon, 0.5)

            # Bounds => each alpha in [alpha_min, alpha_max]
            bounds = [(alpha_min, alpha_max)] * horizon

            res = minimize(cost_function, alpha_init, method='SLSQP',
                           bounds=bounds, options={'maxiter':50, 'ftol':1e-4})
            if not res.success:
                print(f"[WARNING] SLSQP fail at t index {idx}, reason: {res.message}")
                alpha_opt = np.zeros(horizon)
            else:
                alpha_opt = res.x

            # Take the first alpha from the solution
            alpha_k = alpha_opt[0]

            # Step the system using the identified model + noise on v1, v2
            x_next = identified_model_step(x_current, alpha_k, dt)

            # Add random noise to v1, v2
            noise1 = np.random.normal(0, sigma1)
            noise2 = np.random.normal(0, sigma2)
            x_next[1] += dt * noise1
            x_next[3] += dt * noise2

            U_sim.append(alpha_k)
            X_sim.append(x_next)
            x_current = x_next

        X_sim = np.array(X_sim)
        U_sim = np.array(U_sim)

        # Store results
        X_all[s, :len(X_sim), :] = X_sim
        U_all[s, :len(U_sim)] = U_sim

    # Summaries
    X_mean = np.mean(X_all, axis=0)
    X_std  = np.std(X_all, axis=0)
    U_mean = np.mean(U_all, axis=0)
    U_std  = np.std(U_all, axis=0)

    return X_all, U_all, X_mean, X_std, U_mean, U_std
