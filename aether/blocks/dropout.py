import numpy as np
import aether.config as config
from aether.blocks.base import Layer

_philox_dropout_forward = None
_philox_dropout_backward = None 

try:
    import cupy as cp
    
    _philox_dropout_forward = cp.ElementwiseKernel(

    )
except(ImportError, ModuleNotFoundError):
    pass 
class Dropout(Layer): 
    _stream_counter = 0
    def __init__(self, rate, seed=None): 
        super().__init__(seed = None) # from Layer.__init__
        self.keep_rate = 1 - rate

        stream_id = Dropout._stream_counter
        Dropout._stream_counter += 1
        self.seed = self._derive_stream_seed(base_seed=self.seed, stream_id=stream_id)
        self._call_counter = 0 # bumped once per training-mode forward call

        self.forward = self._forward_fallback
        self.backward = self._backward_fallback

    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to map low-level hardware paths."""
        if device == 'cupy' and _philox_dropout_forward is not None and _philox_dropout_backward is not None:
            self.forward = self._forward_gpu
            self.backward = self._backward_gpu
        else:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback

    def _forward_gpu(self, inputs, training):

        if not training:
            self.output = inputs.copy()
            return self.output

        self._call_counter += 1
        offset = self._call_counter
        self.offset = offset

        self.output = _philox_dropout_forward(
            inputs, self.seed, offset, float(self.rate)
        )
        return self.output

    def _backward_gpu(self, dvalues): 

        self.dinputs = _philox_dropout_backward(
            dvalues, self.seed, self.offset, float(self.rate)
        )
        return self.dinputs
    
    def _forward_fallback(self, inputs, training):

        if not training:
            self.output = inputs.copy()

        xp = config.get_array_module(inputs)
        self.binary_mask = xp.random.binomial(1, self.keep_rate, size=inputs.size) \
                            / self.keep_rate
        self.output = inputs * self.binary_mask
        return self.output

    def _backward_fallback(self, dvalues):
        xp = config.get_array_module(dvalues)
        self.dinputs = dvalues * self.binary_mask
        return self.dinputs
        