#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 18:42:04 2023

@author: ghaithalseirawan
"""

import numpy as np

data = np.array([0,1])

# Make a function to initailise weights and biase
def create_parameters():
    init_weights = np.random.normal(0, 1, size = [2])
    init_bias = np.array(0)
    return init_weights, init_bias


def feed_forward(inputs, weights, bias):
    z = np.sum(inputs * weights) + bias
    return z

# Define ReLU activation function
def ReLU(z): 
    return 1 if z >= 0 else 0

#print(ReLU(feed_forward(data, init_weights, init_bias)))

# TRAIN THE PERCEPTRON
# Create function that update the init_weights and init_bias by calculating the error
# which is equal to the difference between the predicted value and the actual value


def update_weights(input_data, predicted_data, target_data, weights, bias, learning_rate):
    error = target_data - predicted_data
    delta_weights = learning_rate * error * input_data
    delta_bias = learning_rate * error
    
    new_weights = weights + delta_weights
    new_bias = bias + delta_bias
    
    return error, new_weights, new_bias

def forward_update(input_data, target_data, weights, bias, learning_rate):
    predicted = ReLU(feed_forward(input_data, weights, bias))
    error, weights, b1 = update_weights(input_data, predicted, target_data, weights, bias, learning_rate)
    return error, weights, b1

# TESTING
#df = np.array([1, -0.5, -2])
#target = 1
#W_ = np.array([0.1, 1.0, 0.5])
#b_ = 0
#alpha = 0.1

#error, W, b = forward_update(df, target, W_, b_, alpha)

#print(error, W, b)


# Train the Perceptron on AND gates

# Create a Mean Absolut Error performance function 
def test_MAE_AND(weights, bias):
    all_possible_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # declare the possible logical gates inputs
    accum_error = 0
    for input_ in all_possible_inputs:
        truth_output = input_[0] and input_[1]
        perceptron_predicted_output = feed_forward(input_, weights, bias)
        error = truth_output - perceptron_predicted_output
        accum_error += np.abs(error)
    return accum_error/all_possible_inputs.shape[0]


# TRAINING CYCLE/ TESTING
learning_rate_ = 0.01
training_cycles = 200
weights, bias = create_parameters()


for cycle in range(training_cycles):
    init_inputs = np.random.binomial(1, 0.5, size=[2])
    target_output = init_inputs[0] and init_inputs[1]
    error, weights, bias = forward_update(data, target_output, weights, bias, learning_rate_)
    if cycle % 10 == 0:
        MEA = test_MAE_AND(weights, bias)
        print(f"Cycle {cycle}, MEA {MEA:,.2f}")

# Why sometimes I'm getting MEA upove 1?

print('Weights after training: '  + str(weights))
print('Bias after training: {:,.2f}'.format(bias))


        
# Train the Perceptron on XOR gates


def test_MAE_XOR(weights, bias):
    all_possible_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # declare the possible logical gates inputs
    accum_error = 0
    for input_ in all_possible_inputs:
        truth_output = input_[0] != input_[1]
        perceptron_predicted_output = feed_forward(input_, weights, bias)
        error = truth_output - perceptron_predicted_output
        accum_error += np.abs(error)
    return accum_error/all_possible_inputs.shape[0]


# TRAINING CYCLE/ TESTING
learning_rate_ = 0.01
training_cycles = 200
weights, bias = create_parameters()


for cycle in range(training_cycles):
    init_inputs = np.random.binomial(1, 0.5, size=[2])
    target_output = init_inputs[0] != init_inputs[1]
    error, weights, bias = forward_update(data, target_output, weights, bias, learning_rate_)
    if cycle % 10 == 0:
        MEA = test_MAE_XOR(weights, bias)
        print(f"Cycle {cycle}, MEA {MEA:,.2f}")


print('Weights after training: '  + str(weights))
print('Bias after training: {:,.2f}'.format(bias))