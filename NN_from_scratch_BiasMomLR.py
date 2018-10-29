'''
=== Neural Network implementation from scratch using Numpy ===
Originally found at:
https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
activation function = sigmoid function
standard feedforward + backpropagation (sgd)
I added momentum, learning rate (constant) and bias
'''

import numpy as np
import matplotlib.pyplot as plt

# sigmoid function definition
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivative function definition
def sigmoid_derivative(x):
    return x * (1 - x)

''' Neural Network Class
Inputbias extends input to each layer with unit biases, bias weights (bias1, bias2) are initialised randomly.
Neural Network weights (weights1, weights2) are initialised randomly.
'''
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.inputbias1 = np.ones((np.shape(self.input)[0], 1)) / np.shape(self.input)[0]
        self.weights1   = np.random.rand(self.input.shape[1], 4)
        self.bias1      = np.random.rand(1, 4)
        self.layer1     = np.zeros((4, 4))
        self.inputbias2 = np.ones((np.shape(self.layer1)[0], 1))
        self.weights2   = np.random.rand(4, 1)
        self.bias2      = np.random.rand(1, 1)
        self.input_mat  = None
        self.weights1_mat = None
        self.layer1_mat = None
        self.weights2_mat = None
        self.y          = y
        self.output     = np.zeros(self.y.shape)        # neural network output to be compared with real y
        self.v_dw1      = 0         # used for momentum calculation
        self.v_dw2      = 0         # used for momentum calculation
        self.alpha      = 0.5       # learning rate
        self.beta       = 0.9       # momentum

        # input signal fed forward through network
    def feedforward(self):
        self.input_mat = np.hstack((self.input, self.inputbias1))
        self.weights1_mat = np.vstack((self.weights1, self.bias1))
        self.layer1 = sigmoid(np.dot(self.input_mat, self.weights1_mat))
        self.layer1_mat = np.hstack((self.layer1, self.inputbias2))
        self.weights2_mat = np.vstack((self.weights2, self.bias2))
        self.output = sigmoid(np.dot(self.layer1_mat, self.weights2_mat))

        # error (y - output) backpropagated through network to tune weights
    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1_mat.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_bias2 = d_weights2[(len(d_weights2))-1:]
        d_weights2 = d_weights2[:4]
        d_weights1 = np.dot(self.input_mat.T,  (np.dot(2*(self.y - self.output) *
                                            sigmoid_derivative(self.output), self.weights2.T) *
                                            sigmoid_derivative(self.layer1)))
        d_bias1 = d_weights1[3:]
        d_weights1 = d_weights1[:3]
        # adding effect of momentum
        self.v_dw1 = (self.beta * self.v_dw1) + ((1 - self.beta) * d_weights1)
        self.v_dw2 = (self.beta * self.v_dw2) + ((1 - self.beta) * d_weights2)
        # update the weights with the derivative (slope) of the loss function
        self.bias1 = self.bias1 + d_bias1
        self.bias2 = self.bias2 + d_bias2
        self.weights1 = self.weights1 + (self.v_dw1 * self.alpha)
        self.weights2 = self.weights2 + (self.v_dw2 * self.alpha)

    # input = simple XOR
if __name__ == "__main__":
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(X, y)

    # iterate through 10000 epochs
    total_loss = []
    for i in range(10000):
        nn.feedforward()
        nn.backprop()
        total_loss.append(sum((nn.y-nn.output)**2))

    iteration_num = list(range(10000))
    # plot loss and print eventual NN outputs
    plt.plot(iteration_num, total_loss)
    plt.show()
    print(nn.output)
