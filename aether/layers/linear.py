import numpy as np
import aether.config as config
from aether.base import Layer
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

        #Gradient on values
        self.dinputs = xp.dot(dvalues, self.weights.T)

    def get_parameters(self):
        #pass We'll let Model call this function 
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
