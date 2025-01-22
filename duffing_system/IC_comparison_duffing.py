import numpy as np
import matplotlib.pyplot as plt

# Import your simulate function
from simulate_2dof_duffing import simulate_true

def check_duffing_trajectories():
    """
    Demonstration script that:
      - Varies alpha1, alpha2 (Duffing cubic parameters),
      - Varies initial conditions,
      - Simulates the 2DOF system,
      - Overlays the trajectories to check for possible chaos or divergence.
    """

    # 1) Global system parameters
    m1, m2 = 1.0, 1.0
    k1, k2 = 1.0, 1.0
    c1, c2 = 0.3, 0.3
    theta_0_1, theta_0_2 = 0.5, 0.5  # or 0.5, etc.

    # 2) Time array
    dt = 0.01
    t_max = 15.0
    t = np.arange(0, t_max, dt)

    # 3) Define a control input array U(t). For simplicity, use zeros here,
    #    or you could try sinusoids or other signals.
    U = np.zeros((len(t), 2))

    # 4) Define some test initial conditions
    #    Each is a 4D state: [x1, v1, x2, v2].
    #    You can add or remove as many as you want.
    initial_conditions = [
        [1.0, 0.0, 0.5, -0.2],       # moderate offset
        [-0.5, 0.3, 1.2, 0.0],        # larger displacement
    ]

    # 5) Vary alpha1, alpha2. E.g. try a few sets.
    alpha_pairs = [
        (2.0, 2.0),
        (5.0, 5.0),
        (8.0, 2.0),
        (8.0, 8.0)
    ]

    # 6) For adding random noise or not. Here, let's do no noise for clarity.
    noise_array_1 = np.zeros(len(t))
    noise_array_2 = np.zeros(len(t))

    # 7) Plot setup
    fig, axs = plt.subplots(
        nrows=len(initial_conditions),
        ncols=2,
        figsize=(12, 3 * len(initial_conditions)),
        sharex=True
    )
    # If there's only one IC, axs might not be 2D, so ensure consistent handling
    if len(initial_conditions) == 1:
        axs = [axs]

    # We will plot x1(t) on the left column and x2(t) on the right column
    # for each initial condition, overlaying different alpha pairs.
    # You can also plot velocities or combine subplots as you wish.

    for row_idx, x0 in enumerate(initial_conditions):
        print(x0)
        # Access subplots
        ax_left = axs[row_idx][0]
        ax_right = axs[row_idx][1]

        for alpha1, alpha2 in alpha_pairs:
            # 8) Simulate for each (alpha1, alpha2)
            X, _ = simulate_true(
                m1, m2,
                c1, c2,
                k1, k2,
                alpha1, alpha2,
                theta_0_1, theta_0_2,
                x0, t,
                U,
                noise_array_1,
                noise_array_2
            )

            # Unpack states
            x1 = X[:, 0]
            x2 = X[:, 2]

            # 9) Plot x1(t) and x2(t)
            label_str = f"α1={alpha1}, α2={alpha2}"
            ax_left.plot(t, x1, label=label_str)
            ax_right.plot(t, x2, label=label_str)

        # Tidy up subplot titles
        ax_left.set_title(f"IC={x0} | x1(t)")
        ax_right.set_title(f"IC={x0} | x2(t)")
        ax_left.set_ylabel("Displacement")
        ax_right.set_ylabel("Displacement")
        ax_left.grid(True)
        ax_right.grid(True)

        # Show legend only on one of the subplots if you prefer
        if row_idx == 0:  # top row
            ax_left.legend()

    # 10) Final formatting
    for col in range(2):
        axs[-1][col].set_xlabel("Time [s]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    check_duffing_trajectories()
