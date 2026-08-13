import weakref

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

        # Per-layer momentum/cache buffers are allocated once and
        # tracked here so update_parameters never re-probes the layer
        # for their existence on every step. 

        self._initialized_layers = weakref.WeakSet()
        # Bias-correction cache -- see _bias_corrections().
        self._bc_cache_iteration = -1
        self._bc1 = None
        self._bc2 = None

        # Pointer-swap targets. _compile_for_device rebinds these to
        # the fused-kernel GPU path when available.
        self.update_parameters = self._update_parameters_fallback

    def _compile_for_device(self, device):
        """
        Triggered by Model.to(device) to bind the fused-kernel GPU path
        or the fallback.
        """
        if device == 'cupy' and _adam_update_kernel is not None:
            self.update_parameters = self._update_parameters_gpu
        else:
            self.update_parameters = self._update_parameters_fallback

    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def post_update_parameters(self):
        self.iterations += 1

    def _bias_corrections(self):
        """
        (1 - beta1**t), (1 - beta2**t) depend only on self.iterations,
        which is constant across every trainable layer within a single
        step (it only advances in post_update_parameters). Cached per
        step instead of recomputing two beta**t calls per layer.
        """
        if self._bc_cache_iteration != self.iterations:
            t = self.iterations + 1
            self._bc1 = 1 - self.beta_1 ** t
            self._bc2 = 1 - self.beta_2 ** t
            self._bc_cache_iteration = self.iterations
        return self._bc1, self._bc2

    def _ensure_layer_state(self, layer, xp):
        """
        One-time momentum/cache allocation for a layer.
        """
        if layer in self._initialized_layers:
            return
        layer.weight_momentums = xp.zeros_like(layer.weights)
        layer.weight_cache = xp.zeros_like(layer.weights)
        layer.bias_momentums = xp.zeros_like(layer.biases)
        layer.bias_cache = xp.zeros_like(layer.biases)
        self._initialized_layers.add(layer)

    @staticmethod
    def _l1_subgradient(param, l1_lambda, xp):
        """
        Sub-gradient of L1 regularization, scaled by l1_lambda: +1 where
        param >= 0, -1 where param < 0 (matching the +1-at-zero
        convention the existing per-layer implementations use).
        """
        return l1_lambda * xp.where(param < 0, -1.0, 1.0).astype(param.dtype)

    def _get_regularized_gradients(self, layer, xp):
        """
        Folds any *coupled* L1/L2 regularization configured on the
        layer itself (weight_regularizer_l1/l2, bias_regularizer_l1/l2)
        into dweights/dbiases. 

        Decoupled AdamW weight decay is NOT part of this method -- that
        never touches the gradient (see _resolve_weight_decay).

        Returns dweights/dbiases unmodified, with no allocation, when
        nothing is configured -- expected to be the common case as
        decoupled weight decay takes over from per-layer L1/L2.

        """
        dweights = layer.dweights
        dbiases = layer.dbiases

        if layer.weight_regularizer_l1 > 0:
            dweights = dweights + self._l1_subgradient(layer.weights, layer.weight_regularizer_l1, xp)
        if layer.weight_regularizer_l2 > 0:
            dweights = dweights + 2 * layer.weight_regularizer_l2 * layer.weights

        if layer.bias_regularizer_l1 > 0:
            dbiases = dbiases + self._l1_subgradient(layer.biases, layer.bias_regularizer_l1, xp)
        if layer.bias_regularizer_l2 > 0:
            dbiases = dbiases + 2 * layer.bias_regularizer_l2 * layer.biases

        return dweights, dbiases

    def _resolve_weight_decay(self, layer):
        """
        Effective decoupled weight-decay coefficient for this layer's
        *weights* this step. Always 0.0 for plain Adam (no
        weight_decay attribute exists on this class at all) and for
        any layer flagged `no_weight_decay`.

        TODO: nothing currently sets `no_weight_decay` -- Batch_Norm
        still lives in the legacy CNN_classes*.py and hasn't been
        touched here since it's out of scope for this pass. Once it's
        migrated, add `no_weight_decay = True` as a class attribute so
        gamma/beta get excluded once AdamW's weight_decay is actually
        turned on -- otherwise this will silently start decaying
        BatchNorm's scale parameter.

        Biases are excluded structurally by the caller, not by this
        flag -- they're never passed through this method.
        """
        if getattr(layer, "no_weight_decay", False):
            return 0.0
        return getattr(self, "weight_decay", 0.0)

    def _apply_weight_decay(self, layer):
        """
        Decoupled L2 weight decay: w -= lr * weight_decay * w, applied
        directly to the parameter and kept out of dweights/the moment
        estimates entirely.
        """
        wd = self._resolve_weight_decay(layer)
        if wd > 0:
            layer.weights -= self.current_learning_rate * wd * layer.weights

    def _update_parameters_fallback(self, layer):
        xp = config.get_array_module(layer.weights)
        self._ensure_layer_state(layer, xp)
        dweights, dbiases = self._get_regularized_gradients(layer, xp)
        bias_correction1, bias_correction2 = self._bias_corrections()

        self._apply_weight_decay(layer)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
            (1 - self.beta_1) * dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
            (1 - self.beta_1) * dbiases

        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * (dweights ** 2)
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * (dbiases ** 2)

        weight_momentums_corrected = layer.weight_momentums / bias_correction1
        bias_momentums_corrected = layer.bias_momentums / bias_correction1
        weight_cache_corrected = layer.weight_cache / bias_correction2
        bias_cache_corrected = layer.bias_cache / bias_correction2

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
            (xp.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
            (xp.sqrt(bias_cache_corrected) + self.epsilon)

    def _update_parameters_gpu(self, layer):
        xp = config.get_array_module(layer.weights)
        self._ensure_layer_state(layer, xp)
        dweights, dbiases = self._get_regularized_gradients(layer, xp)
        
        bc1, bc2 = self._bias_corrections()
        
        w_dtype = layer.weights.dtype
        lr_w = w_dtype.type(self.current_learning_rate)
        beta1_w = w_dtype.type(self.beta_1)
        beta2_w = w_dtype.type(self.beta_2)
        eps_w = w_dtype.type(self.epsilon)
        bc1_w = w_dtype.type(bc1)
        bc2_w = w_dtype.type(bc2)
        wd_w = w_dtype.type(self._resolve_weight_decay(layer))

        dweights = dweights.astype(w_dtype, copy=False)

        _adam_update_kernel(
            layer.weights, dweights, layer.weight_momentums, layer.weight_cache,
            lr_w, beta1_w, beta2_w, eps_w, bc1_w, bc2_w, wd_w,
            layer.weights, layer.weight_momentums, layer.weight_cache,
        )

        #Any layer that keeps a shadow low-precision cast of its weights 
        # needs an explicit nudge
        # here or it'll keep training on a stale cast.
        if hasattr(layer, "invalidate_shadow_caches"):
            layer.invalidate_shadow_caches()

        if getattr(layer, 'biases', None) is not None:
            b_dtype = layer.biases.dtype
            
            # Reuse weight scalar objects if dtypes match (common case)
            if b_dtype == w_dtype:
                lr_b, beta1_b, beta2_b, eps_b, bc1_b, bc2_b = lr_w, beta1_w, beta2_w, eps_w, bc1_w, bc2_w
            else:
                lr_b = b_dtype.type(self.current_learning_rate)
                beta1_b = b_dtype.type(self.beta_1)
                beta2_b = b_dtype.type(self.beta_2)
                eps_b = b_dtype.type(self.epsilon)
                bc1_b = b_dtype.type(bc1)
                bc2_b = b_dtype.type(bc2)

            dbiases = dbiases.astype(b_dtype, copy=False)

            _adam_update_kernel(
                layer.biases, dbiases, layer.bias_momentums, layer.bias_cache,
                lr_b, beta1_b, beta2_b, eps_b, bc1_b, bc2_b, b_dtype.type(0.0),
                layer.biases, layer.bias_momentums, layer.bias_cache,
            )

class AdamW(Adam):
    def __init__(self, learning_rate=.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=.999, weight_decay=0.01):
        
        super().__init__(learning_rate=learning_rate, decay=decay, epsilon=epsilon,
                          beta_1=beta_1, beta_2=beta_2)

        self.weight_decay = weight_decay