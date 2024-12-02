import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Set random seed for reproducibility
# np.random.seed(42)

# 1. Generate Synthetic Data
def generate_data(n_samples=100, noise_std=1.0):
    """
    Generates synthetic data based on a sparse polynomial:
    y = 3 + 2x - x^3 + noise
    Only coefficients for x^0, x^1, and x^3 are non-zero.
    """
    X = np.linspace(-3, 3, n_samples)
    # Define true coefficients (sparse: some coefficients are zero)
    true_coeffs = np.array([3.0, 2.0, 0.0, -1.0])  # y = 3 + 2x - x^3
    # Create polynomial features up to degree 3
    poly = PolynomialFeatures(degree=3, include_bias=True)
    X_poly = poly.fit_transform(X.reshape(-1, 1))
    # Generate target variable with noise
    y = X_poly @ true_coeffs + np.random.normal(0, noise_std, size=n_samples)
    return X, y, true_coeffs

# 2. Prepare Polynomial Features
degree = 3  # Degree of the polynomial
poly = PolynomialFeatures(degree=degree, include_bias=True)

# 3. Fit Models
def fit_models(X, y, alpha=1.0):
    """
    Fits both OLS and Lasso regression models.
    
    Parameters:
    - X: Input feature array.
    - y: Target variable array.
    - alpha: Regularization strength for Lasso.
    
    Returns:
    - models: Dictionary containing fitted models.
    """
    models = {}
    
    # Ordinary Least Squares (OLS)
    ols_pipeline = Pipeline([
        ('poly_features', poly),
        ('linear_regression', LinearRegression())
    ])
    ols_pipeline.fit(X.reshape(-1,1), y)
    models['OLS'] = ols_pipeline
    
    # Lasso Regression with L1 Regularization
    lasso_pipeline = Pipeline([
        ('poly_features', poly),
        ('lasso', Lasso(alpha=alpha, max_iter=10000))
    ])
    lasso_pipeline.fit(X.reshape(-1,1), y)
    models['Lasso'] = lasso_pipeline
    
    return models

# 4. Visualization
def plot_results(X, y, models, true_coeffs):
    """
    Plots the original data, fitted models, and their coefficients.
    
    Parameters:
    - X: Input feature array.
    - y: Target variable array.
    - models: Dictionary containing fitted models.
    - true_coeffs: Array of true coefficients used to generate the data.
    """
    plt.figure(figsize=(14, 6))
    
    # Plot Data and Fitted Curves
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color='gray', label='Data', alpha=0.6)
    
    X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1,1)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    
    for name, model in models.items():
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, label=f'{name} Fit', linewidth=2)
    
    # True Polynomial for Reference
    poly_true = PolynomialFeatures(degree=degree, include_bias=True)
    X_true_poly = poly_true.fit_transform(X_plot)
    y_true = X_true_poly @ true_coeffs
    plt.plot(X_plot, y_true, 'k--', label='True Polynomial', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression with L₁ Regularization')
    plt.legend()
    plt.grid(True)
    
    # Plot Coefficients
    plt.subplot(1, 2, 2)
    coef_names = [f'x^{i}' for i in range(degree+1)]
    true_coeffs_labels = [f'True {name}' for name in coef_names]
    
    # OLS Coefficients
    ols_coeffs = models['OLS'].named_steps['linear_regression'].coef_
    ols_coeffs[0] = models['OLS'].named_steps['linear_regression'].intercept_
    
    # Lasso Coefficients
    lasso_coeffs = models['Lasso'].named_steps['lasso'].coef_
    lasso_coeffs[0] = models['Lasso'].named_steps['lasso'].intercept_
    
    # Plotting
    bar_width = 0.35
    indices = np.arange(degree+1)
    
    plt.bar(indices, ols_coeffs, bar_width, label='OLS', alpha=0.7)
    plt.bar(indices + bar_width, lasso_coeffs, bar_width, label='Lasso', alpha=0.7)
    plt.plot(indices, true_coeffs, 'ro', label='True Coefficients')
    
    plt.xlabel('Polynomial Terms')
    plt.ylabel('Coefficient Value')
    plt.title('Comparison of Coefficients')
    plt.xticks(indices + bar_width / 2, coef_names)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 5. Main Execution
def main():
    # Generate Data
    X, y, true_coeffs = generate_data(n_samples=100, noise_std=5.0)
    
    # Fit Models
    alpha = 0.5  # Regularization strength; adjust to see more/less sparsity
    models = fit_models(X, y, alpha=alpha)
    
    # Plot Results
    plot_results(X, y, models, true_coeffs)
    
    # Print Coefficients
    print("True Coefficients:")
    for i, coef in enumerate(true_coeffs):
        print(f"θ_{i}: {coef}")
    
    print("\nOLS Coefficients:")
    ols_coeffs = models['OLS'].named_steps['linear_regression'].coef_
    ols_coeffs[0] = models['OLS'].named_steps['linear_regression'].intercept_
    for i, coef in enumerate(ols_coeffs):
        print(f"θ_{i}: {coef:.4f}")
    
    print("\nLasso Coefficients:")
    lasso_coeffs = models['Lasso'].named_steps['lasso'].coef_
    lasso_coeffs[0] = models['Lasso'].named_steps['lasso'].intercept_
    for i, coef in enumerate(lasso_coeffs):
        print(f"θ_{i}: {coef:.4f}")

if __name__ == "__main__":
    main()
