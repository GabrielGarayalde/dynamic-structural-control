import pysindy as ps
import numpy as np

def get_initial_guess_from_pysindy(X, X_dot, U, t):
    """
    Use PySINDy to get an initial guess for the model parameters.
    Assume:
    - X is (n_samples, n_states)
    - X_dot is (n_samples, n_states)
    - U is (n_samples,) or (n_samples, n_controls)
    - t is time array.

    Returns:
        initial_guess: a numpy array of parameters [theta_0_1, coeffs_1..., theta_0_2, coeffs_2..., sigma_epsilon_1, sigma_epsilon_2 (optional)]
    """

    # PySINDy setup:
    # If you have a control input, you can include it as a feature in the library.
    # For example, for the 2DOF case:
    # Let's assume you already have a feature library that includes state monomials and control terms.
    # The simplest approach: Let PySINDy guess a model of the form:
    # v1_dot = f1(x1,v1,x2,v2,u)
    # v2_dot = f2(x1,v1,x2,v2,u)
    # and we have no separate noise parameters here (since PySINDy doesn't identify noise terms directly).
    
    # Build feature library similar to what we defined:
    library = ps.PolynomialLibrary(degree=1, include_interaction=True) \
                 + ps.IdentityLibrary()  # Adjust as needed to match features
    # You may need to customize the feature library to match your Bayesian model's features.

    # If U is a single control input, treat it as a single extra dimension
    if U.ndim == 1:
        U = U.reshape(-1,1) 
    
    # Concatenate states and controls for library
    # For PySINDy, typically you provide X and optionally U to create a combined library
    # or you can combine them as an augmented state.
    # Another approach is to pass control inputs as explicit inputs:
    model = ps.SINDy(feature_library=library, feature_names=["x1","v1","x2","v2","u"])
    
    # Fit the model to v1_dot and v2_dot only.
    # PySINDy expects X_dot as derivatives of the states [x1,v1,x2,v2].
    # We have v1_dot = X_dot[:,1] and v2_dot = X_dot[:,3].
    # Let's create a reduced problem where we consider only [v1_dot, v2_dot] as "outputs" and [x1,v1,x2,v2,u] as inputs.
    # One approach: we can fit a multi-output model by giving it the full state and expecting it to rediscover the relationship.
    
    # Actually, SINDy normally fits all state derivatives simultaneously. Here we want it to focus on identifying v1_dot, v2_dot.
    # We can just give it the full X and X_dot and it will return a model for each state derivative.
    # We'll then extract the coefficients for v1_dot and v2_dot from the learned model.

    # Fit the full system:
    model.fit(X, t=t, x_dot=X_dot, u=U) 

    # After fitting, model.coefficients() returns a matrix of shape (n_features, n_output_features)
    # n_output_features = number of state derivatives = 4 for (x1_dot, v1_dot, x2_dot, v2_dot)
    # We know:
    # x1_dot = v1, x2_dot = v2 are trivial. The model might or might not produce them correctly.
    # We really need the rows that correspond to v1_dot and v2_dot.

    # Let's extract them:
    # model.coefficients() is something like shape (n_features, n_states)
    coeff_matrix = model.coefficients()  # shape (n_features, n_states)
    # feature_names = model.get_feature_names() # PySINDy provides the names of features
    
    # The order of states in PySINDy is the same as X: [x1, v1, x2, v2]
    # Derivatives are: X_dot = [x1_dot, v1_dot, x2_dot, v2_dot]
    # Indices: x1_dot = 0, v1_dot = 1, x2_dot = 2, v2_dot = 3
    # We want v1_dot (idx=1) and v2_dot (idx=3)

    # Extract coefficients for v1_dot and v2_dot:
    coeffs_v1 = coeff_matrix[:,1]  # For v1_dot
    coeffs_v2 = coeff_matrix[:,3]  # For v2_dot

    # Now we must align these coefficients with our Bayesian model parameter vector:
    # Bayesian model expects: [theta_0_1, coeffs_1..., theta_0_2, coeffs_2..., sigma_eps_1, sigma_eps_2]
    # We'll assume PySINDy returned a model of form: v1_dot = sum_j c_1j * feature_j, and similarly for v2_dot.
    # Among these features we must identify which corresponds to a constant term (the bias, if any).
    # If the library includes a bias term, it will appear as '1' in feature_names. If not, we might need to add it or handle separately.

    pysindy_feature_names = model.get_feature_names()
    # Find the index of the constant feature if it exists
    # PySINDy uses '1' to represent a constant feature if included by PolynomialLibrary
    if '1' in pysindy_feature_names:
        const_idx = pysindy_feature_names.index('1')
        theta_0_1 = coeffs_v1[const_idx]
        theta_0_2 = coeffs_v2[const_idx]
    else:
        # If no constant term is included in the library, set them manually or to zero.
        theta_0_1 = 0.0
        theta_0_2 = 0.0

    # Remove the constant from the coefficient arrays for matching our format
    # We'll build coeffs_1 and coeffs_2 matching the Bayesian SINDy features order.
    # This requires that the PySINDy library and the Bayesian library are consistent.
    # Let's assume we've matched the same features. If differences exist, you may need a mapping step.

    # Remove the constant feature from PySINDy arrays if present
    if '1' in pysindy_feature_names:
        mask = np.ones_like(coeffs_v1, dtype=bool)
        mask[const_idx] = False
        coeffs_v1 = coeffs_v1[mask]
        coeffs_v2 = coeffs_v2[mask]
        filtered_feature_names = [f for f in pysindy_feature_names if f != '1']
    else:
        filtered_feature_names = pysindy_feature_names[:]

    # Now filtered_feature_names should match get_feature_names() from your Bayesian code.
    # If the order differs, reorder coeffs_v1 and coeffs_v2 accordingly.
    # For simplicity, assume they match directly here.
    # initial guess for sigma:
    sigma_epsilon_1 = 0.1
    sigma_epsilon_2 = 0.1

    # Construct the initial guess parameter vector
    initial_guess = [theta_0_1] + list(coeffs_v1) + [theta_0_2] + list(coeffs_v2) + [sigma_epsilon_1, sigma_epsilon_2]
    return np.array(initial_guess)
