#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:25:02 2023

@author: ghaithalseirawan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()

mnis_train_df = "/data/mnist_train.csv"
mnis_test_df = "/data/mnist_test.csv"

try:
    data = pd.read_csv(cwd + mnis_train_df)
except FileNotFoundError:
    print("File not found! Fetching data from" + cwd + mnis_train_df)
except pd.errors.EmptyDataError:
    print("No data")
except pd.errors.ParserError:
    print("Parse error")


# Reshape a split the data into dev and train sets
data = np.array(data) # Convert the dataset to np array
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:2000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[2000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]


def init_params():
   W1 = np.random.rand(10, 784) - 0.5 # initialise the first layer weights
   b1 = np.random.rand(10, 1) - 0.5 # Bias value
   W2 = np.random.rand(10, 10) - 0.5 # initialise the second layer weights
   b2 = np.random.rand(10, 1) - 0.5
   return W1, b1, W2, b2

# Linear Activation function ReLU
def ReLU(Z):
    return np.maximum(Z, 0) # Element-wise check for each value if it is greater or less than zero

# Non-linear Activation function to the output layer. 
# Return multinomial probability. This important for mulit-class classification problem.
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z))
    return A 

    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)



def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    






