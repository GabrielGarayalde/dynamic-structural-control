import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
n = 1                     # Number of observations (for simplicity, using 1)
y_obs = 7                 # Observed data point
sigma = 1.0               # Known standard deviation

# 1. Bayesian Perspective: Likelihood Function p(y | mu) as a function of mu
mu_values = np.linspace(4, 10, 1000)
likelihood_bayes = norm.pdf(y_obs, loc=mu_values, scale=sigma)

# Optional: Define a prior and compute the posterior
# Here, we'll use a Normal prior for mu
mu_prior_mean = 6.0       # Prior mean
mu_prior_std = 2.0        # Prior standard deviation
prior = norm.pdf(mu_values, loc=mu_prior_mean, scale=mu_prior_std)

# Compute the posterior (unnormalized)
posterior_unnorm = likelihood_bayes * prior
# Normalize the posterior
posterior = posterior_unnorm / np.trapz(posterior_unnorm, mu_values)

# 2. Frequentist Perspective: PDF p(y | mu) as a function of y for fixed mu
# Define fixed mu values
mu_fixed_values = [5.0, 7.0, 9.0]
# Define y range
y_values = np.linspace(4, 10, 500)

# Compute PDFs for each fixed mu
pdfs = [norm.pdf(y_values, loc=mu_fixed, scale=sigma) for mu_fixed in mu_fixed_values]

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bayesian Plot: Likelihood and Posterior
axes[0].plot(mu_values, likelihood_bayes, label='Likelihood $p(y=7 \mid \mu)$', color='blue')
axes[0].plot(mu_values, prior, label='Prior $p(\mu)$', color='green', linestyle='--')
axes[0].plot(mu_values, posterior, label='Posterior $p(\mu \mid y=7)$', color='red')
axes[0].axvline(x=y_obs, color='black', linestyle=':', label=r'Observed $y=7$')
axes[0].set_title('Bayesian Perspective')
axes[0].set_xlabel(r'$\mu$ (Mean)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True)

# Frequentist Plot: PDFs for fixed mu values
colors = ['blue', 'orange', 'green']
for idx, (pdf, mu_fixed) in enumerate(zip(pdfs, mu_fixed_values)):
    axes[1].plot(y_values, pdf, label=f'$\mu = {mu_fixed}$', color=colors[idx])
    # Highlight the observed y_obs on each PDF
    axes[1].plot(y_obs, norm.pdf(y_obs, loc=mu_fixed, scale=sigma), marker='o', color=colors[idx])

axes[1].set_title('Frequentist Perspective')
axes[1].set_xlabel('$y$ (Observed Value)')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
