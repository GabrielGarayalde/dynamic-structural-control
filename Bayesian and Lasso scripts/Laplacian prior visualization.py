import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def laplace_prior(x, mu=0, b=1):
    """
    Compute log of Laplace prior probability
    mu: location parameter
    b: scale parameter (diversity)
    """
    return -np.log(2 * b) - np.abs(x - mu) / b

def gaussian_log_likelihood(x, data, sigma=1):
    """
    Compute log of Gaussian likelihood
    """
    return np.sum(stats.norm.logpdf(data, x, sigma))

def log_posterior_unnormalized(x, data, prior_mu=0, prior_b=1, likelihood_sigma=1):
    """
    Compute log of unnormalized posterior combining Laplace prior and Gaussian likelihood
    """
    return laplace_prior(x, prior_mu, prior_b) + gaussian_log_likelihood(x, data, likelihood_sigma)

# Generate some synthetic data
# np.random.seed(42)
true_mean = 1.5
n_samples = 20
data = np.random.normal(true_mean, 10, n_samples)

# Print sample statistics
print(f"Sample mean: {np.mean(data):.3f}")
print(f"Sample std: {np.std(data):.3f}")
print(f"True mean: {true_mean:.3f}")

# Set up grid for parameter space
x_grid = np.linspace(-1, 2, 200)

# Compute prior, likelihood, and posterior in log space
log_prior = np.array([laplace_prior(x, mu=0, b=10) for x in x_grid])
log_likelihood = np.array([gaussian_log_likelihood(x, data, sigma=1) for x in x_grid])
log_posterior = np.array([log_posterior_unnormalized(x, data, prior_mu=0, prior_b=2, likelihood_sigma=1) 
                         for x in x_grid])

# Convert back to probability space for plotting
prior = np.exp(log_prior - np.max(log_prior))
likelihood = np.exp(log_likelihood - np.max(log_likelihood))
posterior = np.exp(log_posterior - np.max(log_posterior))

# Normalize
prior = prior / np.trapz(prior, x_grid)
likelihood = likelihood / np.trapz(likelihood, x_grid)
posterior = posterior / np.trapz(posterior, x_grid)

# Create single plot with all distributions
plt.figure(figsize=(10, 6))
plt.plot(x_grid, prior, 'b--', label='Prior', alpha=0.7)
plt.plot(x_grid, likelihood, 'g--', label='Likelihood', alpha=0.7)
plt.plot(x_grid, posterior, 'r-', label='Posterior', linewidth=2)
plt.axvline(true_mean, color='k', linestyle='--', label='True Mean')
plt.axvline(np.mean(data), color='gray', linestyle=':', label='Sample Mean')

# Find and plot the posterior mode (peak)
posterior_mode = x_grid[np.argmax(posterior)]
plt.axvline(posterior_mode, color='purple', linestyle='-.', label='Posterior Mode')

plt.title('Prior, Likelihood, and Posterior Distributions')
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
# plt.yscale('log')

# Add text box with key values
plt.text(0.02, 0.98, 
         f'True Mean: {true_mean:.3f}\n'
         f'Sample Mean: {np.mean(data):.3f}\n'
         f'Posterior Mode: {posterior_mode:.3f}',
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()