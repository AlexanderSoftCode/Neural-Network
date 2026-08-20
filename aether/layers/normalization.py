import aether.config as config
from aether.base import Layer

class BatchNorm(Layer):
    _precision_exempt: bool = True
    def __init__ (self, epsilon = 1e-5, momentum = 0.9):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum

        self.n_features = None
        self.gamma = None
        self.beta = None
        self.weights = self.gamma
        self.biases = self.beta
        self.running_mean = None
        self.running_var = None

        # backward cache
        self.inputs = None
        self.normalized = None
        self.inv_std = None
        self.batch_mean = None
        self.batch_var = None

        self.weight_regularizer_l1 = 0
        self.weight_regularizer_l2 = 0
        self.bias_regularizer_l1 = 0
        self.bias_regularizer_l2 = 0

    def build(self, input_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        super().build(input_shape)
        
        if self.n_features is None:
            self.n_features = input_shape[-1]
            
        dim = int(self.n_features)
        xp = config.xp

        self.gamma = xp.ones(dim, dtype=xp.float32)
        self.beta = xp.zeros(dim, dtype=xp.float32)
        self.running_mean = xp.zeros(dim, dtype=xp.float32)
        self.running_var = xp.ones(dim, dtype=xp.float32)
        
        self.weights = self.gamma
        self.biases = self.beta
        
        return self.output_shape

    def forward(self, inputs, training):

        xp = config.get_array_module(inputs)
        # Reduce across all leading dimensions except sample S axis.
        axis = tuple(range(inputs.ndim - 1))        
        if training:
            batch_mean = xp.mean(inputs, axis=axis, keepdims=True)
            batch_var = xp.var(inputs, axis=axis, keepdims=True)

            # Update running stats
            self.running_mean = (
                self.momentum * self.running_mean + (1.0 - self.momentum) * batch_mean.squeeze()
            )
            self.running_var = (
                self.momentum * self.running_var + (1.0 - self.momentum) * batch_var.squeeze()
            )

            # Cache intermediate state
            self.inputs = inputs
            self.batch_mean = batch_mean
            self.batch_var = batch_var
            self.inv_std = 1.0 / xp.sqrt(batch_var + self.epsilon)
            self.normalized = (inputs - batch_mean) * self.inv_std

            return self.gamma * self.normalized + self.beta

        inv_std = 1.0 / xp.sqrt(self.running_var + self.epsilon)
        normalized = (inputs - self.running_mean) * inv_std
        return self.gamma * normalized + self.beta

    def backward(self, dvalues):
        xp = config.get_array_module(dvalues)
        axes = tuple(range(dvalues.ndim - 1))
        N_total = self.inputs.size // self.inputs.shape[-1]

        # Gradients with respect to gamma and beta
        self.dweights = xp.sum(dvalues * self.normalized, axis=axes)
        self.dbiases = xp.sum(dvalues, axis=axes)

        dhatx = dvalues * self.gamma
        
        dvar = xp.sum(
            dhatx * (self.inputs - self.batch_mean) * (-0.5) * (self.inv_std ** 3),
            axis=axes,
            keepdims=True
        )

        dmu = xp.sum(dhatx * -self.inv_std, axis=axes, keepdims=True) + dvar * xp.sum(
            -2.0 * (self.inputs - self.batch_mean), axis=axes, keepdims=True
        ) / N_total

        self.dinputs = (
            dhatx * self.inv_std
            + dvar * 2.0 * (self.inputs - self.batch_mean) / N_total
            + dmu / N_total
        )
        return self.dinputs
    
    def get_config(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "momentum": self.momentum,
        }

    def get_parameters(self) -> dict:
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }

    def set_parameters(
        self,
        gamma=None,
        beta=None,
        running_mean=None,
        running_var=None,
        **kwargs
    ):
        if gamma is not None:
            self.gamma = gamma
            self.weights = self.gamma

        if beta is not None:
            self.beta = beta
            self.biases = self.beta

        if running_mean is not None:
            self.running_mean = running_mean

        if running_var is not None:
            self.running_var = running_var