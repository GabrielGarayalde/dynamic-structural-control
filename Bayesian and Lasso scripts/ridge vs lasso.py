import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for linear regression
np.random.seed(0)
m = 5  # Number of data points (m > p => overdetermined)
x = np.linspace(0, 10, m)
beta_true = 2.5  # True slope
noise = np.random.normal(0, 10, m)
y = beta_true * x + noise  # Linear relationship with noise

# Define a range of beta (slope) values for evaluation
beta_vals = np.linspace(-5, 10, 500)

# Define lambda values for regularization
lambda_vals = [0, 1, 100, 200, 400, 1000, 2000]  # Different regularization strengths

# Prepare the plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for Ridge Regression
ax = axes[0]
for lam in lambda_vals:
    objective_vals = []
    for beta in beta_vals:
        residuals = y - beta * x
        RSS = np.sum(residuals ** 2)
        penalty = lam * beta ** 2  # L2 penalty
        objective = RSS + penalty
        objective_vals.append(objective)
    ax.plot(beta_vals, objective_vals, label=f'λ = {lam}')
ax.set_title('Ridge Regularization')
ax.set_xlabel('Slope (β)')
ax.set_ylabel('Regularized Objective Function')
ax.legend()
ax.grid(True)

# Plot for Lasso Regression
ax = axes[1]
for lam in lambda_vals:
    objective_vals = []
    for beta in beta_vals:
        residuals = y - beta * x
        RSS = np.sum(residuals ** 2)
        penalty = lam * np.abs(beta)  # L1 penalty
        objective = RSS + penalty
        objective_vals.append(objective)
    ax.plot(beta_vals, objective_vals, label=f'λ = {lam}')
ax.set_title('Lasso Regularization')
ax.set_xlabel('Slope (β)')
ax.set_ylabel('Regularized Objective Function')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
