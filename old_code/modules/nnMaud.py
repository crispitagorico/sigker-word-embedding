import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tools import brownian
from transformers import AddTime

def cov(X, Y):
    A = X.shape[0]
    B = Y.shape[0]
    M = X.shape[1]
    N = Y.shape[1]

    K = torch.zeros((A, B, M, N)).type(torch.Tensor)
    K[:, :, 0, :] = 1.
    K[:, :, :, 0] = 1.

    # increments
    for i in range(M - 1):
        for j in range(N - 1):
            inc_X = (X[:, i + 1, :] - X[:, i, :])
            inc_Y = (Y[:, j + 1, :] - Y[:, j, :])
            increment = torch.einsum('ik,jk->ij', inc_X, inc_Y)
            K[:, :, i + 1, j + 1] = K[:, :, i + 1, j].clone() + K[:, :, i, j + 1].clone() \
                                    + K[:, :, i,j].clone() * increment.clone() - K[:,:,i,j].clone()
    return K[:, :, -1, -1].T


def RoughMMD(X_train, y_train):
    K1 = cov(X_train, X_train)
    K2 = cov(X_train, y_train)
    K3 = cov(y_train, y_train)
    return (1./(K1.shape[0]**2))*K1.sum() - (2./(K1.shape[0]*K3.shape[0]))*K2.sum() + (1./(K3.shape[0]**2))*K3.sum()


class NaiveNN(nn.Module):
    def __init__(self, width, length):
        super(NaiveNN, self).__init__()

        # Define the output layer
        self.linear1 = nn.Conv1d(in_channels=width, out_channels=width, kernel_size=1)
        self.linear2 = nn.Conv1d(in_channels=width, out_channels=width, kernel_size=1)

    def forward(self, path):
        path = path.transpose(1, 2)
        path = self.linear1(path)
        path = F.relu(path)
        y_pred = self.linear2(path)

        return y_pred.transpose(2, 1)

def runModel():
    model = NaiveNN(width + 1, length)
    learning_rate = 0.01
    num_epochs = 100

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        # Forward pass
        y_pred = model(X_train)

        loss = RoughMMD(y_pred, y_train)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()

    return hist