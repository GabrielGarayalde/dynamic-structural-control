import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameters
n = 10       # Number of coin tosses
y_obs = 7    # Observed number of heads

# 1. Bayesian Perspective: Likelihood Function p(y | theta) as a function of theta
theta_values = np.linspace(0, 1, 1000)
likelihood = binom.pmf(y_obs, n, theta_values)

# 2. Frequentist Perspective: PMF p(y | theta) as a function of y for fixed theta

# Define fixed theta values for the Frequentist side
theta_fixed_values = [0.3, 0.6]

# Prepare y values
y_values = np.arange(0, n+1)

# Compute PMFs for each fixed theta
pmfs = [binom.pmf(y_values, n, theta) for theta in theta_fixed_values]

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bayesian Likelihood Plot
axes[0].plot(theta_values, likelihood, color='blue')
axes[0].set_title('Bayesian Perspective: Likelihood Function')
axes[0].set_xlabel(r'$\theta$ (Probability of Heads)')
axes[0].set_ylabel(r'$p(y=7 \mid \theta)$')
axes[0].axvline(x=y_obs/n, color='red', linestyle='--', label=r'$\theta = 0.7$')
axes[0].legend()
axes[0].grid(True)

# Frequentist PMF Plot
for idx, (pmf, theta_fixed) in enumerate(zip(pmfs, theta_fixed_values)):
    axes[1].bar(y_values + idx*0.2 - 0.1, pmf, width=0.2, alpha=0.7, label=f'θ = {theta_fixed}')
    # Highlight the observed y_obs
    axes[1].bar(y_obs + idx*0.2 - 0.1, pmf[y_obs], width=0.2, color='red', edgecolor='black')

axes[1].set_title('Frequentist Perspective: PMF p(y | θ)')
axes[1].set_xlabel('Number of Heads y')
axes[1].set_ylabel(r'$p(y \mid \theta)$')
axes[1].set_xticks(y_values)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
