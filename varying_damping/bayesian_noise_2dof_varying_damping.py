import numpy as np
import emcee

class BayesianNoiseEstimator:
    def __init__(self, det_sindy_model, rows_for_coeffs=(1,3), n_walkers=16):
        """
        For a 2DOF system with discovered SINDy model,
        we do MCMC to estimate noise levels sigma_epsilon_1, sigma_epsilon_2.
        """
        self.det_sindy_model = det_sindy_model
        self.rows_for_coeffs = rows_for_coeffs
        self.n_walkers = n_walkers
        self.sampler = None
        self.samples_ = None

    def log_prior(self, theta):
        # Suppose theta = [log_sigma1, log_sigma2]
        # Uniform prior on these logs => ...
        if -10 < theta[0] < 2 and -10 < theta[1] < 2:
            return 0.0
        return -np.inf

    def log_likelihood(self, theta, X, X_dot, U, t):
        # Sigma = exp(theta)
        sigma1, sigma2 = np.exp(theta)
        # We'll do a simple Gaussian likelihood, ignoring correlation
        # Residual = model-predicted X_dot vs. measured X_dot
        dt = t[1] - t[0]
        X_dot_pred = self.det_sindy_model.predict(X, u=U.reshape(-1,1))
        # Extract v1_dot and v2_dot columns from X_dot_pred
        v1dot_idx = self.rows_for_coeffs[0]
        v2dot_idx = self.rows_for_coeffs[1]

        resid1 = X_dot[:, v1dot_idx] - X_dot_pred[:, v1dot_idx]
        resid2 = X_dot[:, v2dot_idx] - X_dot_pred[:, v2dot_idx]

        # Gaussian log-likelihood
        n = len(X)
        ll1 = -0.5 * n * np.log(2*np.pi*sigma1**2) - 0.5*np.sum(resid1**2)/(sigma1**2)
        ll2 = -0.5 * n * np.log(2*np.pi*sigma2**2) - 0.5*np.sum(resid2**2)/(sigma2**2)
        return ll1 + ll2

    def log_posterior(self, theta, X, X_dot, U, t):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, X, X_dot, U, t)

    def fit(self, X, X_dot, U, t, n_steps=1000, initial_sigma=(0.1, 0.1)):
        # MCMC sampler with emcee
        p0 = []
        for i in range(self.n_walkers):
            # Start near the log of the initial guess
            p0.append(np.log(initial_sigma) + 0.01*np.random.randn(2))
        p0 = np.array(p0)

        sampler = emcee.EnsembleSampler(
            self.n_walkers,
            2,
            self.log_posterior,
            args=(X, X_dot, U, t)
        )
        sampler.run_mcmc(p0, n_steps, progress=True)
        self.sampler = sampler
        self.samples_ = sampler.get_chain(discard=int(0.3*n_steps), thin=10, flat=True)

    def get_sigma_samples(self):
        return np.exp(self.samples_)
