"2DOF Bayesian MPC"

import numpy as np
from scipy.optimize import minimize

def identified_model_step_estimated(x, u_val, dt, theta_params, feature_names):
    """
    Perform one step of state propagation using identified model parameters.
    Now we have:
    X = [x1, v1, x2, v2]
    and we have two sets of parameters: one for v1_dot and one for v2_dot.
    """
    # Unpack parameters
    # Parameter vector structure:
    # [theta_0_1, coeffs_for_eq1..., theta_0_2, coeffs_for_eq2..., sigma_epsilon_1, sigma_epsilon_2]
    M = len(feature_names)
    theta_0_1 = theta_params[0]
    coeffs_1 = theta_params[1:1+M]
    theta_0_2 = theta_params[1+M]
    coeffs_2 = theta_params[2+M:2+2*M]

    x1, v1, x2, v2 = x
    features = np.array([
        x1, v1, x2, v2,
        # x1**2, v1**2, x2**2, v2**2,
        # x1*v1, x1*x2, x1*v2, v1*x2, v1*v2, x2*v2,
        u_val
    ])
    
    v1_dot = theta_0_1 + np.dot(coeffs_1, features)
    v2_dot = theta_0_2 + np.dot(coeffs_2, features)
    
    x1_next = x1 + dt*v1
    v1_next = v1 + dt*v1_dot
    x2_next = x2 + dt*v2
    v2_next = v2 + dt*v2_dot
    return np.array([x1_next, v1_next, x2_next, v2_next])

def run_bayesian_mpc(model, x0, t, Q, R, N, u_min, u_max, x_ref, n_mpc_samples=50):
    dt = t[1] - t[0]
    feature_names = model.feature_names

    sampled_params = model.samples[np.random.choice(len(model.samples), size=n_mpc_samples, replace=False)]
    X_all = np.zeros((n_mpc_samples, len(t), 4))
    U_all = np.zeros((n_mpc_samples, len(t)-1))

    def mpc_cost(u_sequence, current_state, theta_params, horizon):
        x_pred = current_state.copy()
        cost = 0.0
        for u_k in u_sequence[:horizon]:
            error = x_pred - x_ref
            cost += error.T @ Q @ error
            cost += R * (u_k**2)
            x_pred = identified_model_step_estimated(x_pred, u_k, dt, theta_params, feature_names)
        return cost

    for s in range(n_mpc_samples):
        print("sample: ", s)
        params = sampled_params[s]
        x_current = x0.copy()
        X_sim = [x_current]
        U_sim = []
        for idx in range(len(t)-1):
            print("timestep :", idx)
            remaining_steps = len(t) - idx - 1
            current_horizon = min(N, remaining_steps)

            def cost_function(u_sequence):
                return mpc_cost(u_sequence, x_current, params, current_horizon)

            u_init = np.zeros(current_horizon)
            bounds = [(u_min, u_max)] * current_horizon

            res = minimize(cost_function, u_init, method='SLSQP', bounds=bounds, options={'maxiter':50,'ftol':1e-4})
            u_opt = res.x[0] if res.success else 0.0

            U_sim.append(u_opt)
            x_next = identified_model_step_estimated(x_current, u_opt, dt, params, feature_names)
            X_sim.append(x_next)
            x_current = x_next

        X_sim = np.array(X_sim)
        U_sim = np.array(U_sim)

        X_all[s, :len(X_sim), :] = X_sim
        U_all[s, :len(U_sim)] = U_sim

    X_mean = np.mean(X_all, axis=0)
    X_std = np.std(X_all, axis=0)
    U_mean = np.mean(U_all, axis=0)
    U_std = np.std(U_all, axis=0)

    return X_all, U_all, X_mean, X_std, U_mean, U_std
