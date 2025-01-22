# SINDy & Bayesian MPC Examples

This repository demonstrates **Sparse Identification of Nonlinear Dynamics (SINDy)** applied to various mechanical and dynamical systems. In each case, we generate synthetic data from known equations (with optional noise and forcing), fit a SINDy model, possibly perform **Bayesian noise estimation**, and then run a **Model Predictive Control (MPC)** loop using the discovered model.

Each subfolder is **self-contained**, housing:

- **`simulate_*.py`** scripts that define the ODEs and integrate them to produce ground-truth data.  
- **`sindy_*.py`** scripts for building polynomial libraries, fitting with PySINDy, and comparing the discovered vs. true coefficients.  
- **`bayesian_noise_*.py`** scripts for running MCMC to estimate the noise parameters.  
- **`bayesian_mpc_*.py`** scripts implementing a Model Predictive Control routine (often single-control \(\alpha(t)\) or a multi-control scenario).  
- **`plot_*.py`** scripts with helper functions for visualizing trajectories, phase plots, control signals, posterior histograms, etc.  
- **`main_*.py`** “driver” scripts that tie the above components together. Typically you:  
  1. Simulate the system,  
  2. Fit SINDy,  
  3. Possibly do Bayesian noise estimation,  
  4. Run an MPC example with the discovered model,  
  5. Plot results.

---

## Overview of Each Subfolder

1. **`duffing_system/`**  
   - Demonstrates SINDy on a Duffing oscillator.  
   - Shows how to capture higher-order polynomial nonlinearities.  
   - May include or reference Bayesian noise inference or MPC if desired.

2. **`varying_stiffness/`**  
   - 2DOF mechanical system with **total stiffness** \(k_{\mathrm{total}}\) distributed by a control \(\alpha(t)\).  
   - Demonstrates how PySINDy captures cross-terms like \(\alpha \times x_1\).  
   - Single-control MPC example: minimize displacements or velocities by adjusting \(\alpha(t)\).

3. **`varying_stiffness_sinusoidal_force/`**  
   - Same **varying stiffness** idea, but with an external **sinusoidal forcing** on one of the masses.  
   - Shows two “inputs” in the SINDy library: \(\alpha(t)\) plus the known sinusoidal function \(\sin(\omega t)\).  
   - Bayesian noise + single-control MPC to mitigate forced vibrations.

4. **`varying_damping/`**  
   - 2DOF system with **total damping** \(c_{\mathrm{total}}\) split by \(\alpha(t)\):  
     \(\,c_1(t)=\alpha(t)c_{\mathrm{total}},\;c_2(t)=(1-\alpha(t))c_{\mathrm{total}}\).  
   - Stiffnesses \(k_1,k_2\) remain constant.  
   - Similar approach: SINDy, Bayesian noise, single-control MPC.

5. **`varying_damping_sinusoidal_force/`**  
   - Combination of **varying damping** plus a **sinusoidal forcing** on one mass.  
   - Two “inputs” to SINDy: the control \(\alpha(t)\) and the known sinusoid.  
   - Shows how the discovered model is used for MPC to reduce forced vibrations.

---

## Customizing & Extending

1. **Change the ODE / Forcing**: Modify the `simulate_*.py` scripts to tweak the equations or forcing amplitude/frequency.  
2. **Adjust the Polynomial Library**: In the `sindy_*.py` scripts, set `poly_degree`, `include_interactions`, etc. This changes which candidate features SINDy attempts.  
3. **Edit the Cost Function**: In `bayesian_mpc_*.py`, you’ll find the `mpc_cost(...)` routine. You can penalize states, control effort, or add constraints.  
4. **Multiple Controls**: If you have more than one control parameter, you can feed a multi-column `u` array to PySINDy. The system can discover cross-terms like `alpha_1 * alpha_2`, `alpha_1 * x_1`, etc.  

---

## References

- **SINDy**:
  - *Discovering Governing Equations from Data: Sparse Identification of Nonlinear Dynamics (SINDy)*, Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz.  
  - [PySINDy Documentation](https://pysindy.readthedocs.io/).

- **Bayesian Noise Inference**:
  - We use [emcee](https://emcee.readthedocs.io/en/stable/) for MCMC on the noise parameters \(\sigma\).

- **MPC**:
  - Basic reference for model predictive control logic. We implement simple horizons, cost functions, and bounds, typically with `scipy.optimize.minimize` (SLSQP).

