import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def gaussian_kernel(x, xi, sigma):
    return np.exp(-((x - xi) ** 2) / (2 * sigma**2))


def predict(x, X, Y, sigma):
    # Compute the weights for all points (we compute for all since the dataset is small)
    weights = gaussian_kernel(x, X, sigma)

    # Calculate the weighted average if there are any weights
    if weights.sum() > 0:
        return np.dot(weights, Y) / weights.sum()
    else:
        return (
            np.nan
        )  # It's generally better to return NaN if no points contribute to the average


def mse(Y_true, Y_pred):
    return ((Y_true - Y_pred) ** 2).mean()


def cross_validate(X, Y, sigmas, k=5):
    kf = KFold(n_splits=k)
    avg_mses = []
    for sigma in sigmas:
        mses = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            Y_pred = [predict(x, X_train, Y_train, sigma) for x in X_test]
            mses.append(mse(Y_test, Y_pred))

        avg_mses.append(np.mean(mses))

    return avg_mses


# Example usage:
sigma = 0.5  # Bandwidth of the Gaussian kernel
radius = 3 * sigma  # Radius to consider points within
x_query = 5  # Point at which to predict

np.random.seed(0)
X = np.linspace(0, 10, 100)
Y = np.sin(X) + np.random.normal(0, 0.2, X.size)

# prediction = predict(x_query, X, Y, sigma, radius)
# print("Prediction at x =", x_query, "is", prediction)

# Define a range of bandwidths to test
sigmas = np.linspace(0.1, 1.0, 10)
results = cross_validate(X, Y, sigmas)

# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(sigmas, results, marker="o")
plt.xlabel("Bandwidth (sigma)")
plt.ylabel("Mean Squared Error")
plt.title("Cross-Validation Results for Different Bandwidths")
plt.savefig("cross-val.png")

# Find the best bandwidth
best_sigma = sigmas[np.argmin(results)]
print("Best bandwidth:", best_sigma, "with MSE:", np.min(results))

# Prediction with best bandwidth
Y_pred = [predict(x, X, Y, best_sigma) for x in X]

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(X, Y, 'o', label='Actual Data')
plt.plot(X, Y_pred, 'r-', label='Predictions with Best Sigma')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Kernel Regression Prediction with Best Bandwidth σ = {best_sigma:.2f}')
plt.legend()
plt.savefig("best_pred.png")


# TODO: KD-trees to speed up the computation of the kernel weights
# TODO: can use K-nn for finding nearest points
# TODO: X with higher dimensions?
