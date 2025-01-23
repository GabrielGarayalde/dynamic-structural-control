
# SINDy & Bayesian MPC Examples

This repository demonstrates **Sparse Identification of Nonlinear Dynamics (SINDy)** applied to various mechanical and dynamical systems. In each case, we generate synthetic data from known equations (with optional noise and forcing), fit a SINDy model, optionally perform **Bayesian noise estimation**, and then run a **Model Predictive Control (MPC)** routine using the discovered model.

### Root-Level “Classic” 2DOF Example

At the root of the repository, you have 6 scripts (e.g. `simulate_2dof.py`, `sindy_2dof.py`, etc.) that demonstrate a **“classic” 2DOF system** where each mass can have an external force input. This simpler example includes:

- **`main_2dof.py`**: The main driver to simulate, run SINDy, do Bayesian noise estimation, and run a double-control MPC.
- **`simulate_2dof.py`**: Defines the system ODE for the classic linear 2DOF with constant or time-varying forcing.
- **`sindy_2dof.py`**: Builds a PySINDy model, including polynomial libraries if needed, and compares discovered vs. true equations.
- **`bayesian_noise_2dof.py`**: Bayesian MCMC (via `emcee`) for estimating noise parameters \(\sigma\).
- **`bayesian_mpc_2dof.py`**: Model predictive control routine that can handle two force inputs.
- **`plot_2dof.py`**: Utilities for plotting trajectories, phase portraits, etc.

---

## Subfolder Overviews

1. **`duffing_system/`**
   - Demonstrates a Duffing oscillator example.
   - Shows how to capture higher-order polynomial nonlinearities.
   - May include Bayesian noise inference or MPC if desired.

2. **`varying_stiffness/`**
   - 2DOF system with **total stiffness** distributed by a single control \(\alpha(t)\).
   - Demonstrates cross-terms like \(\alpha \cdot x_1\) in SINDy.
   - Single-control MPC to minimize displacements or velocities.

3. **`varying_stiffness_sinusoidal_force/`**
   - Same **varying stiffness** idea, but now with a **sinusoidal forcing** on one mass.
   - Two “inputs” to SINDy: \(\alpha(t)\) and \(\sin(\omega t)\).
   - Bayesian noise inference + single-control MPC for forced vibrations.

4. **`varying_damping/`**
   - 2DOF system with **total damping** \(c_{\mathrm{total}}\) split by \(\alpha(t)\).
   - Stiffnesses remain constant.
   - Single-control MPC to re-distribute damping.

5. **`varying_damping_sinusoidal_force/`**
   - Same **varying damping** scenario, plus a **sinusoidal forcing** on one mass.
   - Two “inputs” in SINDy: \(\alpha(t)\) and the forcing.
   - Demonstrates how to discover forced dynamics and apply MPC.

---

The script will:

- Simulate the system to produce `X_true, X_dot_true`.
- Fit a SINDy model (possibly with polynomial libraries, cross‐terms, or multiple inputs).
- Prune small coefficients, compare discovered vs. true expansions.
- (Optionally) run Bayesian noise estimation for \(\sigma\).
- Run an MPC routine to demonstrate closed-loop control with the discovered model.
- Plot relevant trajectories, phase spaces, and control signals.

---

## Customizing & Extending

- **Change the ODE / Forcing**: Edit the `simulate_*.py` files to modify the system equations or forcing amplitude/frequency.
- **Adjust the Polynomial Library**: In `sindy_*.py`, set `poly_degree` or `include_interactions` for more or fewer candidate features.
- **Edit the MPC Cost Function**: In `bayesian_mpc_*.py`, look at the `mpc_cost(...)` routine. You can penalize states, control usage, or add constraints.
- **Multiple Controls**: Pass multi-column `u` arrays to PySINDy. The library can discover cross-terms like \(\alpha_1 \cdot x_2\), etc.

---

## References

- **SINDy**:
  - *Discovering Governing Equations from Data: Sparse Identification of Nonlinear Dynamics (SINDy)*, by Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz.
  - [PySINDy Documentation](https://pysindy.readthedocs.io/en/latest/)

- **Bayesian Noise Inference**:
  - [emcee Documentation](https://emcee.readthedocs.io/en/stable/)

- **MPC**:
  - We typically use a simple horizon-based approach, calling `scipy.optimize.minimize` (SLSQP) to solve each MPC step.
