import numpy as np
import aether.config as config
from aether.base import Layer
class Dense(Layer):
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1 = 0,
                 bias_regularizer_l1 = 0, weight_regularizer_l2 = 0,
                 bias_regularizer_l2 = 0):
        self.n_inputs = n_inputs
        self.n_neurons= n_neurons
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

        self.weights = None
        self.biases = None
    def build(self):
        """
        Called once by Model.finalize(). config.xp is guaranteed to be
        correctly set if the user called model.to() beforehand.
        """
        #With He initalization, our fan_in maintains proper variance through layers.
        xp = config.xp
        self.weights = .01 * xp.random.randn(self.n_inputs, self.n_neurons) * \
                    xp.sqrt(2.0 / self.n_inputs)
        self.biases = xp.zeros((1, self.n_neurons))

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

class Flatten:
    def forward(self, inputs, training):
        # Save shape so we can restore it in backward pass
        self.inputs_shape = inputs.shape
        # Flatten all dimensions except batch size
        self.output = inputs.reshape(inputs.shape[0], -1)

        return self.output
    
    def backward(self, dvalues):
        # Reshape gradients back to input shape
        self.dinputs = dvalues.reshape(self.inputs_shape)
