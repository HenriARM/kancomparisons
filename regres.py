# Housing dataset https://www.kaggle.com/datasets/camnugent/california-housing-prices


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing

# Load the Boston Housing dataset
boston = fetch_california_housing()

# Load the Boston Housing dataset
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name="MEDV")

# Select one feature for 1D regression
column = "AveRooms"
# column = "HouseAge"
# column = "MedInc"
X = X[[column]]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Train an MLP model
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, verbose=True
)
mlp.fit(X_train_scaled, y_train_scaled)

# MLP Predictions
y_train_pred_mlp = mlp.predict(X_train_scaled)
y_test_pred_mlp = mlp.predict(X_test_scaled)

# Inverse transform the predictions to the original scale
y_train_pred_mlp_original = scaler_y.inverse_transform(y_train_pred_mlp[:, np.newaxis])
y_test_pred_mlp_original = scaler_y.inverse_transform(y_test_pred_mlp[:, np.newaxis])

# Evaluate the model
train_mse_mlp = mean_squared_error(y_train, y_train_pred_mlp_original)
test_mse_mlp = mean_squared_error(y_test, y_test_pred_mlp_original)

print("MLP Performance:")
print(f"Train MSE: {train_mse_mlp:.2f}")
print(f"Test MSE: {test_mse_mlp:.2f}")

# Visualize the results for MLP
plt.figure(figsize=(10, 5))

X_train = X_train[:100]
y_train = y_train[:100]
y_train_pred_mlp_original = y_train_pred_mlp_original[:100]

# Plot training data
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color="blue", label="Actual")
plt.scatter(
    X_train, y_train_pred_mlp_original, color="red", label="Predicted", alpha=0.5
)
plt.title("MLP - Training Data")
plt.xlabel(column)
plt.ylabel("MEDV")
plt.legend()

X_test = X_test[:100]
y_test = y_test[:100]
y_test_pred_mlp_original = y_test_pred_mlp_original[:100]

# Plot testing data
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.scatter(X_test, y_test_pred_mlp_original, color="red", label="Predicted", alpha=0.5)
plt.title("MLP - Testing Data")
plt.xlabel(column)
plt.ylabel("MEDV")
plt.legend()

plt.tight_layout()

# Save the figure
plt.savefig("mlp_regression_results.png")

# Clear the figure
plt.clf()
plt.close()

# Print epoch-wise train/test loss
train_loss = mlp.loss_curve_
plt.plot(train_loss, label="Train Loss")
plt.title("MLP Train Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Save the epoch loss plot
plt.savefig("mlp_train_loss.png")

# Clear the figure
plt.clf()
plt.close()
