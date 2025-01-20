import numpy as np
import matplotlib.pyplot as plt

from simulate_1dof_augmented import simulate_true_with_hidden_z
import pysindy as ps

def main():
    # ------------------------------
    # A) System + Simulation Params
    # ------------------------------
    m, c, k = 1.0, 0.3, 1.0
    theta_0 = 0.5
    theta_z = 1.0
    mu_z = 0.0
    sigma_z = 0.1

    dt = 0.01
    t = np.arange(0, 10, dt)

    # Hidden-state IC
    x0, v0, z0 = 1.0, 0.0, 0.0

    # Control input: e.g. a small sinusoid
    u = 0.5 * np.sin(2*np.pi * 0.5 * t)

    # ------------------------------
    # B) Simulate TRUE system but only observe x, v
    # ------------------------------
    X_obs, X_dot_obs = simulate_true_with_hidden_z(
        m, c, k,
        theta_0,
        x0, v0, z0,
        t, u,
        theta_z=theta_z,
        mu_z=mu_z,
        sigma_z=sigma_z,
        seed=42
    )
    # X_obs.shape => (N,2), X_dot_obs.shape => (N,2)
    # We do *not* see z or z_dot.

    # ------------------------------
    # C) Plot Observations
    # ------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(t, X_obs[:,0], 'b-', label='x(t)')
    axs[0].set_ylabel("Displacement x(t)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, X_obs[:,1], 'r-', label='v(t)')
    axs[1].set_ylabel("Velocity v(t)")
    axs[1].set_xlabel("Time [s]")
    axs[1].grid(True)
    axs[1].legend()

    plt.suptitle("1DOF with Hidden OU State (Observed x,v only)")
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # D) Attempt SINDy on [x,v] only
    # ------------------------------
    # X_obs => shape (N,2)
    # U => shape (N,) => transform to (N,1) for SINDy
    u_sindy = u.reshape(-1,1)

    # We pass X_dot_obs => partial derivatives for (x,v).
    # We do NOT have z => so SINDy won't see that "z" was actually driving v_dot.
    library = ps.PolynomialLibrary(degree=2, include_interaction=True, include_bias=True)
    optimizer = ps.STLSQ(threshold=0.001, alpha=0.0, max_iter=10000)
    model = ps.SINDy(feature_library=library, optimizer=optimizer)

    # Fit the model
    model.fit(X_obs, t=dt, x_dot=X_dot_obs, u=u_sindy)

    print("========== Learned SINDy Model (Ignoring hidden z) ==========")
    model.print()

    # Possibly you'd see cross terms in v_dot (like x^2, x*v, etc.)
    # that "explain" the effect of the unobserved z(t).

if __name__ == "__main__":
    main()
