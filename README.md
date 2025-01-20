## Overview

This repository demonstrates **Sparse Identification of Nonlinear Dynamics (SINDy)** applied to various mechanical and dynamical systems:

1. **2DOF System (Force Control):**  
   - We treat each mass as having an external force \(u_1(t)\) or \(u_2(t)\).  
   - We use SINDy to discover the system equations from simulated data (with noise).  
   - We optionally perform **Bayesian noise estimation** and **Model Predictive Control (MPC)** using the discovered model.

2. **2DOF System (Varying Stiffness Control):**  
   - Instead of applying forces, we let the control \(\alpha(t)\in[0,1]\) redistribute a *total stiffness* \(`k_total`\).  
   - This requires polynomial “interaction” terms in SINDy (e.g. \(\alpha x_1\)) to learn how stiffness couples with the states.  
   - We also use Bayesian noise estimation and a single-control MPC routine.

3. **Duffing System Example:**  
   - Demonstrates SINDy on the Duffing oscillator, including polynomial libraries that capture higher-order nonlinearities.  
   - (Optional) you can adapt the Bayesian noise or MPC approach to this system if desired.

Throughout the examples, we show how to:
- Generate synthetic data via **ODE solvers** (`simulate_*.py`).
- Fit SINDy models, prune coefficients, and compare them to known ground truth.
- Run **Bayesian noise estimation** (via MCMC) to find posterior distributions of noise levels.
- Implement **MPC** with the discovered models, handling constraints and cost functions.

---

### Key Scripts

- **`main_2dof.py`:**  
  Demonstrates force-controlled 2DOF system. Creates two sinusoidal inputs, simulates, applies SINDy, performs Bayesian noise estimation, and optionally runs MPC.  
- **`main_2dof_varying_stiffness.py`:**  
  Demonstrates stiffness-controlled 2DOF system with a ratio \(\alpha(t)\). Shows how to include \(\alpha\) as a control in SINDy, how to do Bayesian noise estimation, and single-control MPC.  
- **`main_duffing.py`:**  
  Demonstrates a Duffing oscillator example, using a polynomial library to capture the nonlinearity. Possibly includes or references Bayesian or MPC if desired.

---



## File Details

- **`simulate_*.py`:**  
  Contains ODE definitions and numerical integration for generating ground-truth trajectories.  

- **`sindy_*.py`:**  
  Houses the code for building and fitting PySINDy models with polynomial libraries (optionally including cross-terms, bias, etc.).  

- **`bayesian_noise_*.py`:**  
  Defines a class (e.g. `BayesianNoiseEstimator`) that runs MCMC (via `emcee`) to infer posterior distributions of the noise levels \(\sigma\).  

- **`bayesian_mpc_*.py`:**  
  Implements Model Predictive Control using the discovered SINDy model, sampling from the noise posterior if desired.  

- **`plot_*.py`:**  
  Gathers plotting utilities: trajectories, phase portraits, control signals, posterior histograms, etc.

---

## Extending / Customizing

- **Add / Remove Features**: Tweak `poly_degree` or `include_interactions` in the `sindy_*.py` scripts to control the complexity of your SINDy model.  
- **Multiple Controls**: The varying stiffness example uses one control \(\alpha(t)\). You can generalize to two or more by passing multi-dimensional arrays to PySINDy.  
- **Different Cost / Constraints**: In the MPC scripts, you can modify the cost function (e.g. penalize `(\alpha - 0.5)^2` or changes in control) and add constraints on states, etc.

---

## References and Resources

- **SINDy**:  
  - “Discovering Governing Equations from Data: Sparse Identification of Nonlinear Dynamics (SINDy)” by Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz.  
  - [PySINDy Documentation](https://pysindy.readthedocs.io/)

- **Bayesian Inference** / **MCMC**:
  - [emcee Documentation](https://emcee.readthedocs.io/en/stable/)

- **MPC**:
  - Basic references: [Richard M. Murray, “Optimization-Based Control,” Caltech coursework](https://www.cds.caltech.edu/~murray/wiki/ME115b).
