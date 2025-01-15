import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

###############################################################################
# 1) SIMULATE DATA (TRUE SYSTEM) + ADD MEASUREMENT NOISE
###############################################################################
def simulate_data(
    m=1.0, c=0.3, k=1.0,
    theta_0=0.5,
    theta_z=1.0, mu_z=0.0,
    dt=0.01, T=20.0,
    x0=1.0, v0=0.0, z0=0.0,
    u_amp=0.5,           # amplitude of a sinusoidal control, for example
    sigma_z=0.1,         # diffusion magnitude in z
    proc_noise_scale=1e-6,  # small extra process noise if desired
    meas_noise_std=1e-2,  # measurement noise for x, v
    seed=42
):
    """
    Simulate the 3D state [x,v,z], then produce noisy measurements
    of x(t), v(t). The parameter z is hidden in the real system.
    We'll do a simple Euler-Maruyama with random increments on z.
    """
    np.random.seed(seed)

    # Time steps
    t = np.arange(0, T, dt)
    N = len(t)

    # Allocate arrays for the "true" states + derivatives
    X_true = np.zeros((N, 3))  # columns => x, v, z
    X_true[0] = [x0, v0, z0]

    # Simple control input: sinusoid
    u = u_amp * np.sin(2.0 * np.pi * 0.5 * t)

    # Simulate
    for i in range(N - 1):
        x_i, v_i, z_i = X_true[i]
        dW = np.random.normal(0, np.sqrt(dt))  # for the OU in z

        # Euler updates
        # 1) x
        x_dot = v_i
        x_next = x_i + dt*x_dot

        # 2) v
        v_dot = (theta_0 + u[i] - c*v_i - k*x_i)/m + z_i
        v_next = v_i + dt*v_dot

        # 3) z
        # discrete OU drift + diffusion
        z_drift = theta_z*(mu_z - z_i)
        z_next = z_i + dt*z_drift + sigma_z*dW

        # optional small random "process noise" for x,v
        x_next += np.random.normal(0, proc_noise_scale)
        v_next += np.random.normal(0, proc_noise_scale)

        X_true[i+1] = [x_next, v_next, z_next]

    # Build noisy measurements for x,v only
    # measurement noise ~ N(0, meas_noise_std^2)
    meas_noise = np.random.normal(0, meas_noise_std, size=(N, 2))
    X_meas = X_true[:, :2] + meas_noise

    return t, u, X_true, X_meas


###############################################################################
# 2) EXTENDED KALMAN FILTER SETUP
###############################################################################
def f_state_update(X, dt, u, m, c, k, theta_0, theta_z, mu_z):
    """
    Deterministic update for the state vector X=[x,v,z], ignoring noise.
    We'll do a simple Euler step here. 
    """
    x, v, z = X
    # 1) x_{k+1} = x_k + dt * v_k
    x_next = x + dt*v
    # 2) v_{k+1}
    v_dot = (theta_0 + u - c*v - k*x)/m + z
    v_next = v + dt*v_dot
    # 3) z_{k+1}
    z_drift = theta_z*(mu_z - z)
    z_next = z + dt*z_drift

    return np.array([x_next, v_next, z_next])

def f_state_update_with_noise(X, dt, u, m, c, k_stiff, theta_0, theta_z, mu_z, sigma_z=0.1):
    x, v, z = X
    x_next = x + dt*v
    v_dot = (theta_0 + u - c*v - k_stiff*x)/m + z
    v_next = v + dt*v_dot

    z_drift = theta_z*(mu_z - z)
    # Add random increment for z inside the state equation
    # but this is more like an EnKF approach or a "stochastic" predict step:
    dW = np.random.normal(0, np.sqrt(dt))
    z_next = z + dt*z_drift + sigma_z*dW

    return np.array([x_next, v_next, z_next])


def jacobian_F(X, dt, u, m, c, k, theta_0, theta_z, mu_z):
    """
    Jacobian of f_state_update wrt X = [x, v, z].
    We approximate partial derivatives for the Euler step.

    X_{k+1} = X_k + dt*g(X_k) => F = I + dt * d(g)/dX
    Where g_x = v, g_v = [(theta_0 + u - c*v - k*x)/m + z], g_z = theta_z*(mu_z - z).
    """
    x, v, z = X

    # We'll compute partial derivatives of:
    #   g_x = v
    #   g_v = (theta_0 + u - c*v - k*x)/m + z
    #   g_z = theta_z*(mu_z - z)
    dgdx = np.zeros((3,3))  # derivative of g wrt [x,v,z]

    # g_x wrt x,v,z
    dgdx[0,0] = 0.0   # partial of v wrt x
    dgdx[0,1] = 1.0   # partial of v wrt v
    dgdx[0,2] = 0.0   # partial of v wrt z

    # g_v wrt x,v,z
    # g_v = (theta_0 + u - c*v - k*x)/m + z
    dgdx[1,0] = -k/m   # partial wrt x
    dgdx[1,1] = -c/m   # partial wrt v
    dgdx[1,2] = 1.0    # partial wrt z

    # g_z wrt x,v,z
    # g_z = theta_z*(mu_z - z)
    dgdx[2,0] = 0.0
    dgdx[2,1] = 0.0
    dgdx[2,2] = -theta_z  # partial wrt z

    # For Euler step: X_{k+1} = X_k + dt*g(X_k)
    # => F = I + dt * (dgdx)
    F = np.eye(3) + dt * dgdx
    return F


def h_measurement(X):
    """
    Measurement function: we measure [x, v], ignoring z.
    So y = [x, v].
    """
    x, v, z = X
    return np.array([x, v])


def jacobian_H(X):
    """
    Measurement Jacobian wrt X.
    y = [ x, v ]
    => partial wrt x => [1, 0, 0]
               wrt v => [0, 1, 0]
    => shape (2,3)
    """
    H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    return H


###############################################################################
# 3) EKF IMPLEMENTATION
###############################################################################
def extended_kalman_filter(
    t, u,
    X_meas,         # shape (N,2) => [x_meas, v_meas]
    m, c, k, theta_0, theta_z, mu_z,
    Q, R,
    x0_guess=1.0, v0_guess=0.0, z0_guess=0.0
):
    """
    Perform an Extended Kalman Filter to estimate X=[x,v,z] from measurements
    of [x,v] only.

    Parameters
    ----------
    t : array, shape (N,)
    u : array, shape (N,) - control input
    X_meas : array, shape (N,2) - measured x, v
    Q : array, shape (3,3) - process noise covariance
    R : array, shape (2,2) - measurement noise covariance
    x0_guess, v0_guess, z0_guess : initial state estimates

    Returns
    -------
    X_est : ndarray, shape (N,3)
      The EKF state estimates [x_est, v_est, z_est]
    P_est : ndarray, shape (N,3,3)
      The estimate covariance at each time
    """

    N = len(t)
    dt_array = np.diff(t)
    # Initialize
    X_hat = np.array([x0_guess, v0_guess, z0_guess])  # initial guess
    P_hat = np.eye(3)*0.01  # initial covariance guess

    X_est = np.zeros((N,3))
    P_est = np.zeros((N,3,3))
    X_est[0] = X_hat
    P_est[0] = P_hat

    for i in range(N-1):
        dt = dt_array[i]

        # --- 1) PREDICTION
        # State predict (Euler step)
        X_hat_minus = f_state_update(X_hat, dt, u[i], m, c, k, theta_0, theta_z, mu_z)
        # Linearize about X_hat for process Jacobian
        F_k = jacobian_F(X_hat, dt, u[i], m, c, k, theta_0, theta_z, mu_z)
        P_hat_minus = F_k @ P_hat @ F_k.T + Q

        # --- 2) MEASUREMENT UPDATE
        # We measure y_meas = [x_meas, v_meas] at step k+1
        y_meas = X_meas[i+1]

        # Evaluate measurement function at X_hat_minus
        H_k = jacobian_H(X_hat_minus)  # shape (2,3)
        y_hat_minus = h_measurement(X_hat_minus)  # shape (2,)

        # Kalman gain
        S = H_k @ P_hat_minus @ H_k.T + R
        K = P_hat_minus @ H_k.T @ np.linalg.inv(S)  # shape (3,2)

        # Update
        innovation = (y_meas - y_hat_minus)  # shape (2,)
        X_hat_plus = X_hat_minus + K @ innovation
        P_hat_plus = (np.eye(3) - K @ H_k) @ P_hat_minus

        # Save
        X_hat = X_hat_plus
        P_hat = P_hat_plus

        X_est[i+1] = X_hat
        P_est[i+1] = P_hat

    return X_est, P_est


###############################################################################
# 4) DEMO MAIN
###############################################################################
if __name__ == "__main__":
    # ------------------------------
    # A) Generate "true" data
    # ------------------------------
    t, u, X_true, X_meas = simulate_data(
        m=1.0, c=0.3, k=1.0,
        theta_0=0.5,
        theta_z=1.0, mu_z=0.0,
        dt=0.01, T=20.0,
        x0=1.0, v0=0.0, z0=0.0,
        sigma_z=0.1,
        meas_noise_std=0.1,
        seed=42
    )
    # X_true => shape (N,3): columns = [x_true, v_true, z_true]
    # X_meas => shape (N,2): columns = [x_meas, v_meas]

    # ------------------------------
    # B) EKF Parameters
    # ------------------------------
    # Process noise covariance Q (3x3), 
    #   interpret it as some guess about random disturbances in [x, v, z].
    Q = np.diag([1e-6, 1e-6, 1e-4])
    Q = np.diag([1e-7, 1e-7, 1e-5])
    # Measurement noise covariance R for [x, v]
    R = np.diag([1e-4, 1e-4])
    # R = np.diag([1e-3, 1e-3])

    # ------------------------------
    # C) Run EKF
    # ------------------------------
    X_est, P_est = extended_kalman_filter(
        t, u,
        X_meas,
        m=1.0, c=0.3, k=1.0, theta_0=0.5,
        theta_z=1.0, mu_z=0.0,
        Q=Q, R=R,
        x0_guess=1.1,  # slightly off initial guesses
        v0_guess=-0.1,
        z0_guess=0.2
    )
    # X_est => shape (N,3): [x_est, v_est, z_est]

    # ------------------------------
    # D) Plot Results
    # ------------------------------
    fig, axs = plt.subplots(3,1, figsize=(10,8))

    # 1) x(t)
    axs[0].plot(t, X_true[:,0], 'k-', label='True x(t)')
    axs[0].plot(t, X_meas[:,0], 'rx', alpha=0.3, label='Measured x(t)')
    axs[0].plot(t, X_est[:,0], 'b--', label='EKF x_est')
    axs[0].set_ylabel('x(t)')
    axs[0].grid(True)
    axs[0].legend()

    # 2) v(t)
    axs[1].plot(t, X_true[:,1], 'k-', label='True v(t)')
    axs[1].plot(t, X_meas[:,1], 'rx', alpha=0.3, label='Measured v(t)')
    axs[1].plot(t, X_est[:,1], 'b--', label='EKF v_est')
    axs[1].set_ylabel('v(t)')
    axs[1].grid(True)
    axs[1].legend()

    # 3) z(t) - hidden forcing
    axs[2].plot(t, X_true[:,2], 'k-', label='True z(t)')
    axs[2].plot(t, X_est[:,2], 'b--', label='EKF z_est')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('z(t)')
    axs[2].grid(True)
    axs[2].legend()

    plt.suptitle("Extended Kalman Filter: Estimating Hidden OU Force z(t)")
    plt.tight_layout()
    plt.show()

    ############################################################################
    # 5) (NEW) USE PY-SINDY ON EKF RESULTS
    ############################################################################

    # A) If you only want [x,v] in SINDy:
    X_ekf_2D = X_est[:, :2]  # shape (N,2)
    # Let PySINDy do the derivative internally (or we can do finite diff).
    # Control input => shape (N,1)
    u_reshaped = u.reshape(-1,1)

    # Create a SINDy model (polynomial library, etc.)
    model_2D = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=3, include_interaction=True),
        optimizer=ps.STLSQ(threshold=0.8)
    )
    dt = 0.01  # same as in simulation
    model_2D.fit(X_ekf_2D, t=dt, u=u_reshaped)

    print("\n===== SINDy model ignoring hidden z =====")
    model_2D.print()

    # # B) If you want the 3D system so that SINDy sees z:
    X_ekf_3D = X_est  # shape (N,3)
    # Optionally pass in the same control input:
    # For a 3D system, we can do:
    model_3D = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=3, include_interaction=True),
        optimizer=ps.STLSQ(threshold=1)
    )
    model_3D.fit(X_ekf_3D, t=dt, u=u_reshaped)

    print("\n===== SINDy model with z included =====")
    model_3D.print()
    
    
    # Suppose X_ekf_2D = np.column_stack([x_est, v_est])
    # v_dot_modified as computed above
    # Also reshape your control u if needed:
    u_for_sindy = u.reshape(-1, 1)
    
    # Create your SINDy model:
    model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=2, include_interaction=True),
        optimizer=ps.STLSQ(threshold=0.8)
    )
    
    # SINDy can either:
    # (1) do its own derivative on X_ekf_2D if you don't provide v_dot_modified
    # (2) or you explicitly pass x_dot and v_dot
    
    # Because we want to specifically pass v_dot_modified for the second equation,
    # we can manually pass x_dot_est for the first equation and v_dot_modified for the second.
    # Let's assume x_dot_est = v_est (from the EKF).
    x_est = X_est[:, 0]  # shape (N,)
    v_est = X_est[:, 1]  # shape (N,)
    z_est = X_est[:, 2]  # shape (N,)
    
    dt = t[1] - t[0]
    v_dot_est = np.gradient(v_est, dt)

    
    v_dot_modified = v_dot_est - z_est

    x_dot_est = v_est  # since x_dot = v in 1DOF
    data_Xdot = np.column_stack([x_dot_est, v_dot_modified])
    
    # Fit:
    model.fit(X_ekf_2D, t=dt, x_dot=data_Xdot, u=u_for_sindy)
    
    model.print()