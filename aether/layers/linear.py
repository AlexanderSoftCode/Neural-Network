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

        self.precision_policy = config.DTypePolicy(compute_dtype = None)

        # Ephemeral forward cache for backward reuse
        # Is also not used during inference
        self._inputs_compute = None
        self._weights_compute = None 

    def build(self, seed: int | None = None):
        """
        Called once by Model.finalize(). config.xp is guaranteed to be
        correctly set if the user called model.to() beforehand.
        """
        #With He initalization, our fan_in maintains proper variance through layers.
        xp = config.xp
        std = xp.sqrt(2.0 / self.n_inputs, dtype=xp.float32)

        if seed is not None:
            rng = xp.random.RandomState(seed)
            raw_weights = rng.randn(self.n_inputs, self.n_neurons)
        else:
            raw_weights = xp.random.randn(self.n_inputs, self.n_neurons)

        self.weights = raw_weights.astype(xp.float32, copy=False) * std
        self.biases = xp.zeros((1, self.n_neurons), dtype=xp.float32)

    def _apply_precision(self, policy):
        """
        Called on Model.set_precision(), stores current policy and invalidates shadow
        cache. All astype checks happen on forward.
        """
        self.precision_policy = policy or config.DTypePolicy()

    def forward(self, inputs, training):
        xp = config.get_array_module(inputs)

        inputs_c, weights_c, biases_c = self.precision_policy.cast_to_compute(
            inputs, self.weights, self.biases
        )

        if training:
            self.inputs = inputs
            self._inputs_compute = inputs_c
            self._weights_compute = weights_c
        else:
            self.inputs = None
            self._inputs_compute = None
            self._weights_compute = None

        return xp.dot(inputs_c, weights_c) + biases_c 
    
    def backward(self, dvalues):

        xp = config.get_array_module(dvalues)

        if self._inputs_compute is None or self._weights_compute is None:
            raise RuntimeError(
                "Dense.backard() called without a preceeding training=True during" \
                "forward pass. If you are manually tracing through the layers, please" \
                "add training=True to Dense.forward()"
            )
        dvalues_c = self.precision_policy.cast_to_compute(dvalues)

        #Fallback, rederive from self.inputs/self.weights rather then crashing
        inputs_c = self._inputs_compute if self._inputs_compute is not None \
        else self.precision_policy.cast_to_compute(self.inputs)[0]
        weights_c = self._weights_compute if self._weights_compute is not None \
        else self.precision_policy.cast_to_compute(self.weights)[0]

        dweights = xp.dot(self.inputs.T, dvalues)
        dbiases = xp.sum(dvalues, axis = 0, keepdims = True)
        dinputs_c = xp.dot(dvalues_c, weights_c.T)

        self.dweights, self.dbiases = self.precision_policy.cast_to_param(dweights, dbiases)
        self.dinputs = dinputs_c.astype(self.inputs.dtype, copy=False)

        self._inputs_compute = None
        self._weights_compute = None

        return self.dinputs
    
    def get_parameters(self):
        #pass We'll let Model call this function 
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

class Flatten(Layer):
    def forward(self, inputs, training):
        xp = config.get_array_module(inputs)

        if training:
            self.inputs_shape = inputs.shape

        return xp.ascontiguousarray(inputs.reshape(inputs.shape[0], -1))
    
    def backward(self, dvalues):
        # Reshape gradients back to input shape
        self.dinputs = dvalues.reshape(self.inputs_shape)

        return self.dinputs