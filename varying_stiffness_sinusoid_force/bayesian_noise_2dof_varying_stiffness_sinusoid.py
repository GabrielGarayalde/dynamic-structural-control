import numpy as np
import emcee

class BayesianNoiseEstimator:
    def __init__(self, det_sindy_model, rows_for_coeffs=(1,3), n_walkers=16):
        """
        Bayesian MCMC to estimate sigma_epsilon_1, sigma_epsilon_2
        in a 2DOF system with alpha(t) + sinusoidal forcing.
        """
        self.det_sindy_model = det_sindy_model
        self.rows_for_coeffs = rows_for_coeffs
        self.n_walkers = n_walkers
        self.sampler = None
        self.samples_ = None

    def log_prior(self, theta):
        # theta = [log_sigma1, log_sigma2]
        # uniform prior in some range
        if -10 < theta[0] < 2 and -10 < theta[1] < 2:
            return 0.0
        return -np.inf

    def log_likelihood(self, theta, X, X_dot, U_2d, t):
        sigma1, sigma2 = np.exp(theta)
        dt = t[1] - t[0]

        # Predict the derivatives from the discovered model
        X_dot_pred = self.det_sindy_model.predict(X, u=U_2d)

        v1dot_idx = self.rows_for_coeffs[0]
        v2dot_idx = self.rows_for_coeffs[1]
        resid1 = X_dot[:, v1dot_idx] - X_dot_pred[:, v1dot_idx]
        resid2 = X_dot[:, v2dot_idx] - X_dot_pred[:, v2dot_idx]

        n = len(X)
        ll1 = -0.5 * n*np.log(2*np.pi*sigma1**2) - 0.5*np.sum(resid1**2)/(sigma1**2)
        ll2 = -0.5 * n*np.log(2*np.pi*sigma2**2) - 0.5*np.sum(resid2**2)/(sigma2**2)
        return ll1 + ll2

    def log_posterior(self, theta, X, X_dot, U_2d, t):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, X, X_dot, U_2d, t)

    def fit(self, X, X_dot,
            alpha_array, sin_forcing_array, t,
            n_steps=1000, initial_sigma=(0.1, 0.1)):
        U_2d = np.column_stack([alpha_array, sin_forcing_array])
        p0 = []
        for i in range(self.n_walkers):
            p0.append(np.log(initial_sigma) + 0.01*np.random.randn(2))
        p0 = np.array(p0)

        sampler = emcee.EnsembleSampler(
            self.n_walkers,
            2,
            self.log_posterior,
            args=(X, X_dot, U_2d, t)
        )
        sampler.run_mcmc(p0, n_steps, progress=True)
        self.sampler = sampler
        self.samples_ = sampler.get_chain(discard=int(0.3*n_steps), thin=5, flat=True)

    def get_sigma_samples(self):
        return np.exp(self.samples_)
