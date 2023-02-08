#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 18:27:54 2023

@author: ghaithalseirawan
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)

inputs = np.array([
    [0,0,1], 
    [1,1,1], 
    [1,0,1], 
    [0,1,1]
])

outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)
weights = 2 * np.random.random((3,1)) - 1

for iteration in range(10000):
    inputs_layer = inputs
    outputs_layer = sigmoid(np.dot(inputs_layer, weights))
    error = outputs - outputs_layer
    adjustments = error * sigmoid_derivative(outputs_layer)
    weights += np.dot(inputs_layer.T, adjustments)

print("Output after training:")
print(outputs_layer)