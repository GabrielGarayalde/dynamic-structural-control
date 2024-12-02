import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso

# Set random seed for reproducibility
np.random.seed(0)

# Parameters
m = 5  # Number of data points
beta_true = 1  # True slope
sigma = 0.1  # Noise standard deviation

# Generate random predictor data
x = np.linspace(0, 10, m)
noise = np.random.normal(0, sigma, m)
y = beta_true * x + noise

# Define beta range and lambda values
beta_vals = np.linspace(-1, 3, 500)
lambda_vals = [0, 20, 100, 500]

# Compute RSS
residuals_matrix = y[:, np.newaxis] - beta_vals * x[:, np.newaxis]
RSS = np.sum(residuals_matrix**2, axis=0)

# Compute regularization terms and objectives
ridge_penalties = {}
ridge_objectives = {}
lasso_penalties = {}
lasso_objectives = {}

for lam in lambda_vals:
    ridge_penalties[lam] = lam * beta_vals**2
    ridge_objectives[lam] = RSS + ridge_penalties[lam]
    
    lasso_penalty = lam * np.abs(beta_vals)
    lasso_penalties[lam] = lasso_penalty
    lasso_objectives[lam] = RSS + lasso_penalty

# Create 4 subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Ridge Regularization Terms with RSS overlay
ax = axes[0, 0]
ax.plot(beta_vals, RSS, color='black', label='RSS', linewidth=2, linestyle='--')
for lam in lambda_vals:
    ax.plot(beta_vals, ridge_penalties[lam], label=f'λ = {lam}')
ax.set_title('Ridge: RSS and Regularization Terms')
ax.set_xlabel('Slope (β)')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True)

# Ridge Total Objective
ax = axes[1, 0]
for lam in lambda_vals:
    ax.plot(beta_vals, ridge_objectives[lam], label=f'λ = {lam}')
ax.set_title('Ridge: Total Objective Function')
ax.set_xlabel('Slope (β)')
ax.set_ylabel('Objective Function')
ax.legend()
ax.grid(True)

# Lasso Regularization Terms with RSS overlay
ax = axes[0, 1]
ax.plot(beta_vals, RSS, color='black', label='RSS', linewidth=2, linestyle='--')
for lam in lambda_vals:
    ax.plot(beta_vals, lasso_penalties[lam], label=f'λ = {lam}')
ax.set_title('Lasso: RSS and Regularization Terms')
ax.set_xlabel('Slope (β)')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True)

# Lasso Total Objective
ax = axes[1, 1]
for lam in lambda_vals:
    ax.plot(beta_vals, lasso_objectives[lam], label=f'λ = {lam}')
ax.set_title('Lasso: Total Objective Function')
ax.set_xlabel('Slope (β)')
ax.set_ylabel('Objective Function')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# Print the beta value that minimizes RSS
min_rss_beta = beta_vals[np.argmin(RSS)]
print(f"\nBeta value that minimizes RSS: {min_rss_beta:.3f}")
print(f"Minimum RSS value: {np.min(RSS):.3f}")