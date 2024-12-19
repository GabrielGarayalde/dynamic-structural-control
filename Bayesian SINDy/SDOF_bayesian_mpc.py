import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

# ------------------------------
# 5. Model Predictive Control with Bayesian Uncertainty
# ------------------------------
def identified_model_step_estimated(x, u_val, dt, theta_0, coeffs_v):
    """
    Perform one step of state propagation using the identified model parameters.
    """
    x_current, v_current = x
    # Compute derivatives
    # Assuming features are [x, v, x^2, v^2, x*v, u]
    features = np.array([x_current, v_current, x_current**2, v_current**2, x_current * v_current, u_val])
    v_dot = theta_0 + np.dot(coeffs_v, features)
    # Update states using Euler's method
    x_next = x_current + dt * v_current
    v_next = v_current + dt * v_dot
    return np.array([x_next, v_next])


def run_bayesian_mpc(model, x0, t, Q, R, N, u_min, u_max, x_ref, n_mpc_samples=50):
    """
    Run MPC with uncertainty by sampling from the posterior.
    For each sampled parameter set, run MPC and gather trajectories.
    """
    dt = t[1] - t[0]

    # Draw parameter samples
    sampled_params = model.samples[np.random.choice(len(model.samples), size=n_mpc_samples, replace=False)]

    # Arrays to store results
    X_all = np.zeros((n_mpc_samples, len(t), 2))
    U_all = np.zeros((n_mpc_samples, len(t)-1))  # len(t)-1 because control is applied at each step except the last

    # MPC cost function
    def mpc_cost(u_sequence, current_state, theta_0_id, coeffs_v_id, horizon):
        # Compute cost over the prediction horizon
        x_pred = current_state.copy()
        cost = 0.0
        for u_k in u_sequence[:horizon]:
            error = x_pred - x_ref
            cost += error.T @ Q @ error
            cost += R * (u_k ** 2)
            # Predict next state
            x_pred = identified_model_step_estimated(x_pred, u_k, dt, theta_0_id, coeffs_v_id)
        return cost

    print("Running Bayesian MPC simulations...")
    for s in tqdm(range(n_mpc_samples), desc="MPC Sim"):
        params = sampled_params[s]
        theta_0_id = params[0]
        coeffs_v_id = params[1:-1]

        x_current = x0.copy()
        X_sim = [x_current]
        U_sim = []
        for idx in range(len(t)-1):
            # Determine remaining steps
            remaining_steps = len(t) - idx - 1
            current_horizon = min(N, remaining_steps)

            # Define MPC optimization problem
            def cost_function(u_sequence):
                return mpc_cost(u_sequence, x_current, theta_0_id, coeffs_v_id, current_horizon)

            # Initial guess for control sequence
            u_init = np.zeros(current_horizon)

            # Bounds for control inputs
            bounds = [(u_min, u_max)] * current_horizon

            # Optimize control sequence
            res = minimize(
                cost_function,
                u_init,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-4}
            )

            if res.success:
                u_opt = res.x[0]
            else:
                u_opt = 0.0  # Fallback control
                print(f"Optimization failed at time {t[idx]:.2f}")

            U_sim.append(u_opt)
            x_next = identified_model_step_estimated(x_current, u_opt, dt, theta_0_id, coeffs_v_id)
            X_sim.append(x_next)

            # Update current state
            x_current = x_next

        X_sim = np.array(X_sim)
        U_sim = np.array(U_sim)

        X_all[s, :len(X_sim), :] = X_sim
        U_all[s, :len(U_sim)] = U_sim

    # Compute statistics
    X_mean = np.mean(X_all, axis=0)
    X_std = np.std(X_all, axis=0)
    U_mean = np.mean(U_all, axis=0)
    U_std = np.std(U_all, axis=0)

    return X_all, U_all, X_mean, X_std, U_mean, U_std