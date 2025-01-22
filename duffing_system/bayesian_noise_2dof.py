# bayesian_noise_2dof.py

import numpy as np
import emcee

class BayesianNoiseEstimator:
    """
    Bayesian approach estimating the noise std devs (sigma_epsilon_1, sigma_epsilon_2)
    assuming the deterministic system is already identified by a SINDy model.
    """
    def __init__(self, det_sindy_model, rows_for_coeffs=(1,3), n_walkers=16):
        self.model = det_sindy_model
        self.rows_for_coeffs = rows_for_coeffs
        self.n_walkers = n_walkers
        self.samples = None

    def log_prior(self, theta):
        """
        Half-Cauchy(0,1) prior for sigma1, sigma2 => must be > 0
        """
        sigma1, sigma2 = theta
        if sigma1 <= 0 or sigma2 <= 0:
            return -np.inf
        scale = 1.0
        logp1 = -np.log(np.pi*scale*(1+(sigma1/scale)**2))
        logp2 = -np.log(np.pi*scale*(1+(sigma2/scale)**2))
        return logp1 + logp2

    def log_likelihood(self, theta, X, X_dot, U=None, t=None):
        """
        Compare v1_dot, v2_dot predictions from SINDy model with actual X_dot.
        """
        sigma1, sigma2 = theta

        # SINDy "predict" typically gives x_dot if fit with x_dot = f(X,U).
        X_dot_pred = self.model.predict(X, u=U)

        v1_dot_true = X_dot[:, self.rows_for_coeffs[0]]
        v2_dot_true = X_dot[:, self.rows_for_coeffs[1]]
        v1_dot_pred = X_dot_pred[:, self.rows_for_coeffs[0]]
        v2_dot_pred = X_dot_pred[:, self.rows_for_coeffs[1]]

        def normal_loglike(y, y_pred, sigma):
            return -0.5 * np.sum(((y - y_pred) / sigma)**2
                                 + np.log(2*np.pi*sigma**2))

        ll1 = normal_loglike(v1_dot_true, v1_dot_pred, sigma1)
        ll2 = normal_loglike(v2_dot_true, v2_dot_pred, sigma2)
        return ll1 + ll2

    def log_probability(self, theta, X, X_dot, U=None, t=None):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta, X, X_dot, U=U, t=t)
        return lp + ll

    def fit(self, X, X_dot, U=None, t=None, n_steps=2000, initial_sigma=(0.1, 0.1)):
        """
        MCMC to estimate sigma_epsilon_1, sigma_epsilon_2.
        """
        n_params = 2
        pos = np.zeros((self.n_walkers, n_params))
        pos[:, 0] = initial_sigma[0] + 0.01 * np.random.randn(self.n_walkers)
        pos[:, 1] = initial_sigma[1] + 0.01 * np.random.randn(self.n_walkers)
        pos = np.abs(pos)  # ensure positivity

        sampler = emcee.EnsembleSampler(
            self.n_walkers, n_params, self.log_probability,
            args=(X, X_dot, U, t)
        )

        # Burn-in
        state = sampler.run_mcmc(pos, 100, progress=True)
        sampler.reset()

        # Production
        sampler.run_mcmc(state, n_steps, progress=True)

        # Flatten chain
        self.samples = sampler.get_chain(discard=100, thin=10, flat=True)
        return self

    def get_sigma_samples(self):
        """
        Retrieve posterior samples [sigma1, sigma2].
        """
        if self.samples is None:
            raise ValueError("Model not fitted yet.")
        return self.samples
