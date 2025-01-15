import numpy as np
import pysindy as ps
from pysindy.feature_library import CustomLibrary
import pandas as pd

###############################################################################
# 1) Comparison function that includes feature names in the table
###############################################################################
def compare_coeffs(true_coeffs, initial_guess, feature_names):
    """
    Compare true coefficients with initial guesses by calculating absolute and
    percentage differences, labeling each entry by its corresponding feature name.

    Parameters
    ----------
    true_coeffs : list or array-like
        The list of true coefficient values.
    initial_guess : list or array-like
        The list of coefficient values estimated by PySINDy (or any model).
    feature_names : list or array-like
        Names corresponding to each coefficient in the same order.

    Returns
    -------
    comparison_df : pd.DataFrame
        DataFrame comparing true coeffs vs. initial guess, with differences.
    """
    # Convert to arrays
    true_coeffs = np.array(true_coeffs).flatten()
    initial_guess = np.array(initial_guess).flatten()
    feature_names = np.array(feature_names).flatten()

    # Basic checks
    if not (len(true_coeffs) == len(initial_guess) == len(feature_names)):
        raise ValueError(
            "Lengths of true_coeffs, initial_guess, and feature_names must match."
        )

    # Build columns
    abs_diff = np.abs(true_coeffs - initial_guess)
    # Handle zero-division in percentage difference
    pct_diff = []
    for tval, ad in zip(true_coeffs, abs_diff):
        if tval != 0:
            pct_diff.append(100.0 * ad / np.abs(tval))
        else:
            pct_diff.append(np.nan)  # or define as np.inf, etc.

    comparison_df = pd.DataFrame({
        "Feature": feature_names,
        "True Coeff.": true_coeffs,
        "Initial Guess": initial_guess,
        "Absolute Diff": abs_diff,
        "% Difference": pct_diff
    })

    return comparison_df


###############################################################################
# 2) Main function to build and fit a PySINDy model, then produce a table
###############################################################################
def get_initial_guess_from_pysindy(
    X,
    X_dot,
    U,
    t,
    true_coeffs,
    # Which terms to include:
    include_bias=True,
    include_linear=True,
    include_quadratic=True,
    include_interactions=True,
    # For naming the variables in final features (for clarity only):
    variable_names=('x1', 'v1', 'x2', 'v2', 'u'),
    # Noise standard deviations (appended at the end of the parameter vector):
    sigma_epsilon_1=0.1,
    sigma_epsilon_2=0.1,
):
    """
    Build a PySINDy model using a custom library, fit it to data (X, X_dot, U),
    extract an initial guess for your parameters, and compare to 'true_coeffs'.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 4)
        State data, columns = [x1, v1, x2, v2].
    X_dot : np.ndarray of shape (n_samples, 4)
        Time derivatives, columns = [x1_dot, v1_dot, x2_dot, v2_dot].
    U : np.ndarray of shape (n_samples, 1) or (n_samples,)
        Control input for second mass (optional).
    t : np.ndarray of shape (n_samples,)
        Time array for the data samples.
    true_coeffs : list or array-like
        The list of "true" coefficient values to compare against.
        Must have the same length as the final parameter vector we build.
    include_bias : bool
        Whether to include a constant (bias) term in the library.
    include_linear : bool
        Whether to include linear terms in each variable: x, v, etc.
    include_quadratic : bool
        Whether to include x^2, v^2, etc.
    include_interactions : bool
        Whether to include pairwise interaction terms x1*x2, etc.
    variable_names : tuple of str
        Names for your columns in X (+ U). Used for labeling the library.
    sigma_epsilon_1 : float
        Noise parameter appended at the end of the final parameter vector.
    sigma_epsilon_2 : float
        Noise parameter appended at the end of the final parameter vector.

    Returns
    -------
    initial_guess : np.ndarray
        A 1D array containing:
          [theta_0_1, (all v1_dot coefficients), theta_0_2, (all v2_dot coefficients),
           sigma_epsilon_1, sigma_epsilon_2].
    """
    # 1) Reshape U if it is 1D
    if U is not None and U.ndim == 1:
        U = U.reshape(-1, 1)

    # 2) Combine X and U so that library sees [x1, v1, x2, v2, u] as features
    X_all = np.hstack((X, U)) if U is not None else X

    print("Shapes inside get_initial_guess_from_pysindy:")
    print("  X_all:", X_all.shape, "(N x 5 if U is included, else N x 4)")
    print("  X_dot:", X_dot.shape, "(N x 4 for [x1_dot, v1_dot, x2_dot, v2_dot])")
    print("  t:", t.shape)

    # 3) Prepare library functions
    library_functions = []
    function_names = []

    # If bias included, CustomLibrary automatically adds "1".
    # We do not manually add a bias function.

    if include_linear:
        library_functions.append(lambda x: x)
        function_names.append(lambda s: s)  # e.g. "x1"

    if include_quadratic:
        library_functions.append(lambda x: x**2)
        function_names.append(lambda s: f"{s}^2")

    if include_interactions:
        library_functions.append(lambda x, y: x * y)
        function_names.append(lambda s1, s2: f"{s1}*{s2}")

    # 4) Create the CustomLibrary
    custom_library = CustomLibrary(
        library_functions=library_functions,
        function_names=function_names,
        interaction_only=False,  # so we allow var1*var1 if the library does that
        include_bias=include_bias
    )
    # Let PySINDy know how to name the base variables:
    custom_library.variable_names = list(variable_names)

    # 5) Create a SINDy model using that library and fit
    model = ps.SINDy(feature_library=custom_library)
    model.fit(X, t=t, x_dot=X_dot, u=U)

    # 6) Extract the coefficient matrix
    # shape = (n_states, n_features_generated_by_library)
    coeff_matrix = model.coefficients()
    
    # We'll see something like:
    #   row 0 => x1_dot
    #   row 1 => v1_dot
    #   row 2 => x2_dot
    #   row 3 => v2_dot
    print("\nSINDy feature names (in PySINDy order):")
    feature_names_sindy = model.get_feature_names()  # includes '1' if bias
    print(feature_names_sindy)
    print("Coefficient matrix shape:", coeff_matrix.shape)

    # 7) Extract the row for v1_dot and v2_dot
    v1_dot_coeffs = coeff_matrix[1, :]  # row 1 => v1_dot
    v2_dot_coeffs = coeff_matrix[3, :]  # row 3 => v2_dot

    # 8) Identify the bias if it exists
    #    PySINDy typically calls it "1" in feature_names_sindy
    try:
        bias_idx = feature_names_sindy.index("1")
        theta_0_1 = v1_dot_coeffs[bias_idx]
        theta_0_2 = v2_dot_coeffs[bias_idx]
    except ValueError:
        # No bias in the library
        bias_idx = None
        theta_0_1 = 0.0
        theta_0_2 = 0.0

    # 9) We want to exclude the bias from the "regular" part of the feature vector
    #    The features that remain are everything in feature_names_sindy except "1".
    #    We'll keep them in the exact order PySINDy gave them (just skipping the bias).
    def extract_coeffs_excluding_bias(all_coeffs, all_feats, bias_name="1"):
        """Return array of coefficients excluding the bias term, in PySINDy order."""
        results = []
        for feat, cval in zip(all_feats, all_coeffs):
            if feat == bias_name:
                continue
            results.append(cval)
        return np.array(results)

    v1_dot_nonbias = extract_coeffs_excluding_bias(v1_dot_coeffs, feature_names_sindy, bias_name="1")
    v2_dot_nonbias = extract_coeffs_excluding_bias(v2_dot_coeffs, feature_names_sindy, bias_name="1")

    # 10) Build the final initial_guess vector
    #     Format: [theta_0_1, v1_dot_coeffs..., theta_0_2, v2_dot_coeffs..., sigma_epsilon_1, sigma_epsilon_2]
    initial_guess = np.concatenate([
        [theta_0_1],
        v1_dot_nonbias,
        [theta_0_2],
        v2_dot_nonbias,
        [sigma_epsilon_1, sigma_epsilon_2]
    ])

    # 11) Build the matching feature names, also in that order
    #     1) "theta_0_1"
    #     2) For each non-bias feature (in PySINDy order), prefix with "v1_dot_"
    #     3) "theta_0_2"
    #     4) For each non-bias feature (in PySINDy order), prefix with "v2_dot_"
    #     5) "sigma_epsilon_1", "sigma_epsilon_2"
    nonbias_feats = [f for f in feature_names_sindy if f != "1"]

    # Optionally, rename each SINDy feature in a more “math-like” format, or just keep them as-is
    # e.g. x1^2 => "x1^2", x1*x2 => "x1*x2", etc. It's already done if you used 'variable_names'.
    v1_names = [f"v1_dot_{f}" for f in nonbias_feats]
    v2_names = [f"v2_dot_{f}" for f in nonbias_feats]

    feature_names_final = (
        ["theta_0_1"] +
        v1_names +
        ["theta_0_2"] +
        v2_names +
        ["sigma_epsilon_1", "sigma_epsilon_2"]
    )

    # 12) Compare with the true_coeffs, building a table
    comparison_table = compare_coeffs(
        true_coeffs,
        initial_guess,
        feature_names_final
    )

    # 13) Print the results
    pd.set_option('display.float_format', lambda x: f'{x:.6f}')  # nicer float formatting
    print("\nComparison Table:")
    print(comparison_table)

    return initial_guess
