import numpy as np
import aether.config as config
from aether.custom_kernels.adam_kernel import _adam_update_kernel

# General starting learning rate for SGD is 1.0, with a decay down to 0.1. For Adam, a good starting 
# LR is 0.001 (1e-3), decaying down to 0.0001 (1e-4). Different problems may require different 
# values here, but these are decent to start.
class Adam:
    def __init__(self, learning_rate=.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2  # used to be known as our rho

        self.layers = []
        self._step_impl = self._step_fallback

    def init_params(self, trainable_layers):
        """
        Called once during Model.finalize() to register trainable layers
        and pre-allocate optimizer and momentum cache buffers in fp32.
        """
        self.layers = trainable_layers
        xp = config.xp

        for layer in self.layers:
            layer.weight_momentums = xp.zeros_like(layer.weights, dtype=xp.float32)
            layer.weight_cache = xp.zeros_like(layer.weights, dtype=xp.float32)

            if getattr(layer, 'biases', None) is not None:
                layer.bias_momentums = xp.zeros_like(layer.biases, dtype=xp.float32)
                layer.bias_cache = xp.zeros_like(layer.biases, dtype=xp.float32)

    def _compile_for_device(self, device):
        """
        Triggered by Model.to(device) to bind the fused-kernel GPU path
        or the fallback.
        """
        if device == 'cupy' and _adam_update_kernel is not None:
            self._step_impl = self._step_gpu
        else:
            self._step_impl = self._step_fallback


    def step(self):
        """Unified optimizer entry point executed once per training step"""
        if self.decay:
            self.current_learning_rate = np.float32(
                self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
            )
        t = self.iterations + 1
        bias_correction_1 = np.float32(1.0 - (self.beta_1 ** t))
        bias_correction_2 = np.float32(1.0 - (self.beta_2 ** t))

        self._step_impl(bias_correction_1, bias_correction_2)

        self.iterations += 1
    @staticmethod
    def _l1_subgradient(param, l1_lambda, xp):
        """Sub-gradient of L1 regularization matching the +1-at-zero convention."""
        return l1_lambda * xp.where(param < 0, -1.0, 1.0).astype(param.dtype)

    def _get_regularized_gradients(self, layer, xp):
        """
        Folds any *coupled* L1/L2 regularization configured on the
        layer itself (weight_regularizer_l1/l2, bias_regularizer_l1/l2)
        into dweights/dbiases. 
        """
        dweights = layer.dweights
        dbiases = layer.dbiases

        if layer.weight_regularizer_l1 > 0:
            dweights = dweights + self._l1_subgradient(layer.weights, layer.weight_regularizer_l1, xp)
        if layer.weight_regularizer_l2 > 0:
            dweights = dweights + 2.0 * layer.weight_regularizer_l2 * layer.weights

        if layer.bias_regularizer_l1 > 0:
            dbiases = dbiases + self._l1_subgradient(layer.biases, layer.bias_regularizer_l1, xp)
        if layer.bias_regularizer_l2 > 0:
            dbiases = dbiases + 2.0 * layer.bias_regularizer_l2 * layer.biases

        return dweights, dbiases

    def _resolve_weight_decay(self, layer):
        """Resolves decoupled weight decay coefficient (AdamW)."""
        if getattr(layer, "no_weight_decay", False):
            return 0.0
        return getattr(self, "weight_decay", 0.0)

    def _step_gpu(self, bias_correction_1, bias_correction_2):
        """Fused CUDA / CuPy elementwise kernel path."""
        xp = config.xp

        # Cast all scalar hyperparameters to float32 once per step
        learning_rate_scalar = np.float32(self.current_learning_rate)
        beta_1_scalar = np.float32(self.beta_1)
        beta_2_scalar = np.float32(self.beta_2)
        epsilon_scalar = np.float32(self.epsilon)
        bias_correction_1_scalar = np.float32(bias_correction_1)
        bias_correction_2_scalar = np.float32(bias_correction_2)
        zero_weight_decay_scalar = np.float32(0.0)

        for layer in self.layers:
            dweights, dbiases = self._get_regularized_gradients(layer, xp)
            weight_decay_scalar = np.float32(self._resolve_weight_decay(layer))

            # Fused update for weights
            _adam_update_kernel(
                layer.weights,
                dweights,
                layer.weight_momentums,
                layer.weight_cache,
                learning_rate_scalar,
                beta_1_scalar,
                beta_2_scalar,
                epsilon_scalar,
                bias_correction_1_scalar,
                bias_correction_2_scalar,
                weight_decay_scalar,
                layer.weights,
                layer.weight_momentums,
                layer.weight_cache,
            )

            # Fused update for biases (weight decay is always 0.0 for biases)
            if getattr(layer, 'biases', None) is not None and dbiases is not None:
                _adam_update_kernel(
                    layer.biases,
                    dbiases,
                    layer.bias_momentums,
                    layer.bias_cache,
                    learning_rate_scalar,
                    beta_1_scalar,
                    beta_2_scalar,
                    epsilon_scalar,
                    bias_correction_1_scalar,
                    bias_correction_2_scalar,
                    zero_weight_decay_scalar,
                    layer.biases,
                    layer.bias_momentums,
                    layer.bias_cache,
                )

            # Invalidate any low-precision compute casts stored on the layer
            if hasattr(layer, "invalidate_shadow_caches"):
                layer.invalidate_shadow_caches()

    def _step_fallback(self, bias_correction_1, bias_correction_2):
        """CPU / NumPy vectorized update path."""
        xp = config.xp
        learning_rate = np.float32(self.current_learning_rate)
        epsilon = np.float32(self.epsilon)
        beta_1 = np.float32(self.beta_1)
        beta_2 = np.float32(self.beta_2)
        one_minus_beta_1 = np.float32(1.0) - beta_1
        one_minus_beta_2 = np.float32(1.0) - beta_2

        for layer in self.layers:
            dweights, dbiases = self._get_regularized_gradients(layer, xp)

            # Decoupled weight decay (AdamW)
            weight_decay = np.float32(self._resolve_weight_decay(layer))
            if weight_decay > 0.0:
                layer.weights -= learning_rate * weight_decay * layer.weights

            # Update weight momentums and second moment cache
            layer.weight_momentums = beta_1 * layer.weight_momentums + one_minus_beta_1 * dweights
            layer.weight_cache = beta_2 * layer.weight_cache + one_minus_beta_2 * (dweights ** 2)

            weight_momentums_corrected = layer.weight_momentums / bias_correction_1
            weight_cache_corrected = layer.weight_cache / bias_correction_2

            layer.weights -= learning_rate * weight_momentums_corrected / (
                xp.sqrt(weight_cache_corrected) + epsilon
            )

            # Update bias momentums and cache (if biases are present)
            if dbiases is not None:
                layer.bias_momentums = beta_1 * layer.bias_momentums + one_minus_beta_1 * dbiases
                layer.bias_cache = beta_2 * layer.bias_cache + one_minus_beta_2 * (dbiases ** 2)

                bias_momentums_corrected = layer.bias_momentums / bias_correction_1
                bias_cache_corrected = layer.bias_cache / bias_correction_2

                layer.biases -= learning_rate * bias_momentums_corrected / (
                    xp.sqrt(bias_cache_corrected) + epsilon
                )

            # Invalidate any low-precision compute casts stored on the layer
            if hasattr(layer, "invalidate_shadow_caches"):
                layer.invalidate_shadow_caches()
                
class AdamW(Adam):
    def __init__(self, learning_rate=.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=.999, weight_decay=0.01):
        
        super().__init__(learning_rate=learning_rate, decay=decay, epsilon=epsilon,
                          beta_1=beta_1, beta_2=beta_2)

        self.weight_decay = weight_decay