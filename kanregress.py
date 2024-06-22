import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# Load the Boston Housing dataset
boston = fetch_california_housing()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='MEDV')

column = "MedInc"
X = X[[column, "HouseAge"]]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Convert the data to PyTorch tensors
train_input = torch.tensor(X_train_scaled, dtype=torch.float32)
train_label = torch.tensor(y_train_scaled, dtype=torch.float32)
test_input = torch.tensor(X_test_scaled, dtype=torch.float32)
test_label = torch.tensor(y_test_scaled, dtype=torch.float32)

# Create the dataset dictionary
dataset = {
    'train_input': train_input,
    'train_label': train_label,
    'test_input': test_input,
    'test_label': test_label
}

# # Save the dataset to a file (optional)
# # torch.save(dataset, 'boston_housing_dataset.pt')

from kan import KAN
import torch

# TODO: grid, k
# TODO: MLP size was like 100
model = KAN(width=[2, 1, 1], grid=3, k=3)


# from kan.utils  create_dataset
# f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
# dataset = create_dataset(f, n_var=2)
model.train(dataset, opt="LBFGS", steps=20)


# from kan import KAN
# import torch
# model = KAN(width=[2,3,2,1])
# x = torch.normal(0,1,size=(100,2))
# model(x)
# beta = 100
# model.plot(beta=beta)
# plt.savefig("pic.png")

# TODO: plot loss
# TODO: plot pred
# TODO: add reqs and publish
