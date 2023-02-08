# neural_network_from_scratch
Building a Neural Network to classify handwritten digits (MNIST database) without using with out Tensorflow or Pytorch, using only math and numpy. 

The code is for a 2-layer neural network that performs multi-class classification. The network architecture consists of an input layer, a hidden layer, and an output layer. The input layer has 784 neurons, the hidden layer has 10 neurons, and the output layer has 10 neurons, each corresponding to one of the 10 classes.

## Requirements
python3
numpy
pandas
matplotlib

## Data

The code uses MNIST data. Downloaded from https://pjreddie.com/projects/mnist-in-csv/


# The code does the following:

- Reads in the MNIST training and test data as pandas dataframes
- Preprocesses the data into numpy arrays and splits them into training and dev sets.
- Initializes the parameters for the neural network (weights and biases for the two layers)
- Does forward propagation to get the output from the network
- Does backward propagation to update the weights and biases
- Repeats the forward and backward propagation for a set number of iterations
- Gets predictions from the network using the final weights and biases
- Computes accuracy by comparing the predictions with the actual labels
- The code can be run by simply executing the file "app.py".

## Code details

The `init_params()` function initializes the weights and biases of the first and second layers. The weights are initialized to random values between -0.5 and 0.5 using the np.random.rand function.

The `ReLU` function is the activation function for the hidden layer and returns the element-wise maximum between the input Z and 0. The softmax function is the activation function for the output layer and returns the multi-nominal probability.

The `forward_prop` function implements forward propagation, which computes the activation of each layer given the input X and the weights and biases. It returns the pre-activations Z1 and Z2 and the activations A1 and A2 of the hidden layer and output layer, respectively.

The `ReLU_deriv` function calculates the derivative of the ReLU activation function. The one_hot function converts a target Y from integer-valued to one-hot encoded form.

The `backward_prop` function implements backward propagation, which computes the gradients of the weights and biases with respect to the loss. It returns the gradients dW1, db1, dW2, and db2.

The `update_params` function updates the weights and biases based on the computed gradients and a learning rate alpha.

- The activation functions used in the network are ReLU (rectified linear unit) for the hidden layer and softmax for the output layer.
- The code uses the cross-entropy loss to train the network.
- The code updates the parameters using gradient descent with a fixed learning rate (alpha).

## Acknowledgment

Special thanks to Samson Zhang for his presenting this code. https://www.samsonzhang.com/

## License

This Notebook has been released under the Apache 2.0 open source license.