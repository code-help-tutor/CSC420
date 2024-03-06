WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
######################################
# Assignement 2 for CSC420
# MNIST clasification example
# Author: Jun Gao
######################################

import numpy as np

def cross_entropy_loss_function(prediction, label):
    #TODO: compute the cross entropy loss function between the prediction and ground truth label.
    # prediction: the output of a neural network after softmax. It can be an Nxd matrix, where N is the number of samples,
    #           and d is the number of different categories
    # label: The ground truth labels, it can be a vector with length N, and each element in this vector stores the ground truth category for each sample.
    # Note: we take the average among N different samples to get the final loss.
    pass

def sigmoid(x):
    # TODO: compute the softmax with the input x: y = 1 / (1 + exp(-x))
    pass

def softmax(x):
    # TODO: compute the softmax function with input x.
    #  Suppose x is Nxd matrix, and we do softmax across the last dimention of it.
    #  For each row of this matrix, we compute x_{j, i} = exp(x_{j, i}) / \sum_{k=1}^d exp(x_{j, k})
    pass

class OneLayerNN():
    def __init__(self, num_input_unit, num_output_unit):
        #TODO: Random Initliaize the weight matrixs for a one-layer MLP.
        # the number of units in each layer is specified in the arguments
        # Note: We recommend using np.random.randn() to initialize the weight matrix: zero mean and the variance equals to 1
        #       and initialize the bias matrix as full zero using np.zeros()
        pass

    def forward(self, input_x):
        #TODO: Compute the output of this neural network with the given input.
        # Suppose input_x is an Nxd matrix, where N is the number of samples and d is the number of dimension for each sample.
        # Compute output: z = softmax (input_x * W_1 + b_1), where W_1, b_1 are weights, biases for this layer
        # Note: If we only have one layer in the whole model and we want to use it to do classification,
        #       then we directly apply softmax **without** using sigmoid (or relu) activation
        pass

    def backpropagation_with_gradient_descent(self, loss, learning_rate, input_x, label):
        #TODO: given the computed loss (a scalar value), compute the gradient from loss into the weight matrix and running gradient descent
        # Note that you may need to store some intermidiate value when you do forward pass, such that you don't need to recompute them
        # Suggestions: you need to first write down the math for the gradient, then implement it to compute the gradient
        pass

# [Bonus points] This is not necessary for this assignment
class TwoLayerNN():
    def ___init__(self, num_input_unit, num_hidden_unit, num_output_unit):
        #TODO: Random Initliaize the weight matrixs for a two-layer MLP with sigmoid activation,
        # the number of units in each layer is specified in the arguments
        # Note: We recommend using np.random.randn() to initialize the weight matrix: zero mean and the variance equals to 1
        #       and initialize the bias matrix as full zero using np.zeros()
        pass

    def forward(self, input_x):
        #TODO: Compute the output of this neural network with the given input.
        # Suppose input_x is Nxd matrix, where N is the number of samples and d is the number of dimension for each sample.
        # Compute: first layer: z = sigmoid (input_x * W_1 + b_1) # W_1, b_1 are weights, biases for the first layer
        # Compute: second layer: o = softmax (z * W_2 + b_2) # W_2, b_2 are weights, biases for the second layer
        pass

    def backpropagation_with_gradient_descent(self, loss, learning_rate):
        #TODO: given the computed loss (a scalar value), compute the gradient from loss into the weight matrix and running gradient descent
        # Note that you may need to store some intermidiate value when you do forward pass, such that you don't need to recompute them
        # Suggestions: you need to first write down the math for the gradient, then implement it to compute the gradient
        pass
