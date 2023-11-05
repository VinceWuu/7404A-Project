import numpy as np
from scipy.optimize import minimize
from numpy.linalg import norm, svd

# Define the logistic regression prediction function
def logistic_regression(W, X):
    return 1 / (1 + np.exp(-np.dot(W, X.T)))

# Define the prediction matrix G(W) (could be the logistic regression function itself or other suitable form)
def G(W, X_pos):
    return logistic_regression(W, X_pos)

# Define the objective function incorporating F
def objective_function(WF_flat, X_pos, X_neg, C1, C2, lambda1, lambda2, n_features, n_samples):
    W = WF_flat[:n_samples * n_features].reshape(n_samples, n_features)
    F = WF_flat[n_samples * n_features:].reshape(n_samples, n_samples)

    # Calculate logistic losses as before
    logistic_loss_pos = np.sum(np.log(1 + np.exp(-np.dot(W, X_pos.T))))
    logistic_loss_neg = np.sum(np.log(1 + np.exp(np.dot(W, X_neg.T))))

    # Calculate Frobenius norm squared of W
    frobenius_norm_W = norm(W, 'fro')**2

    # Calculate nuclear norm of F (sum of singular values)
    nuclear_norm_F = np.sum(svd(F, compute_uv=False))

    # Calculate Frobenius norm squared of the difference between F and G(W)
    frobenius_norm_diff = norm(F - G(W, X_pos), 'fro')**2

    # Combine losses with regularization terms
    loss = frobenius_norm_W + C1 * logistic_loss_pos + C2 * logistic_loss_neg \
           + lambda1 * nuclear_norm_F + lambda2 * frobenius_norm_diff

    return loss

# Example data
n_samples = 10
n_features = 5
n_neg = 20
X_pos = np.random.randn(n_samples, n_features)
X_neg = np.random.randn(n_neg, n_features)
initial_W = np.random.randn(n_samples, n_features)
initial_F = np.random.randn(n_samples, n_samples)  # Initialize F

# Regularization parameters
C1 = 1
C2 = 1
lambda1 = 0.1  # Regularization parameter for nuclear norm of F
lambda2 = 0.1  # Regularization parameter for Frobenius norm of (F - G(W))

# Flatten W and F for the optimizer
initial_WF_flat = np.concatenate([initial_W.flatten(), initial_F.flatten()])

# Run the minimization
res = minimize(
    fun=objective_function,
    x0=initial_WF_flat,
    args=(X_pos, X_neg, C1, C2, lambda1, lambda2, n_features, n_samples),
    method='L-BFGS-B'
)

# Extract W and F from the optimization result
optimized_W = res.x[:n_samples * n_features].reshape(n_samples, n_features)
optimized_F = res.x[n_samples * n_features:].reshape(n_samples, n_samples)

print("Optimized Weight Matrix W:")
print(optimized_W)
print("Optimized Intermediate Matrix F:")
print(optimized_F)
