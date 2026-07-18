import numpy as np
import aether.config as config
from aether.blocks.base import Layer
class Dense(Layer):
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1 = 0,
                 bias_regularizer_l1 = 0, weight_regularizer_l2 = 0,
                 bias_regularizer_l2 = 0):
        xp = config.xp
        #With He initalization, our fan_in maintains proper variance through layers.
        self.weights = .01 * xp.random.randn(n_inputs, n_neurons) * \
            xp.sqrt(2.0 / n_inputs)
        self.biases = xp.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
    def forward(self, inputs, training):

        xp = config.get_array_module(inputs)
        self.inputs = inputs 
        self.output = xp.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):

        xp = config.get_array_module(dvalues)
        self.dweights = xp.dot(self.inputs.T, dvalues)
        self.dbiases = xp.sum(dvalues, axis = 0, keepdims = True)

        if self.weight_regularizer_l1 > 0:
             dL1 = xp.ones_like(self.weights)
             dL1 [self.weights < 0] = -1
             self.dweights += self.weight_regularizer_l1 * dL1
        
        if self.weight_regularizer_l2 > 0:
             self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
             dL1 = xp.ones_like(self.biases)
             dL1 [self.biases < 0 ] = -1
             self.dbiases += self.bias_regularizer_l1 * dL1 
        
        if self.bias_regularizer_l2 > 0:
             self.dbiases += 2* self.bias_regularizer_l2 * self.biases

        #Gradient on values
        self.dinputs = xp.dot(dvalues, self.weights.T)

    def get_parameters(self):
        #pass We'll let Model call this function 
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
