import numpy as np
import aether.config as config
from aether.base import Layer
from aether.custom_kernels import dropout as gpu_dropout

class Dropout(Layer): 
    _stream_counter = 0
    def __init__(self, rate, seed=None): 
        super().__init__(seed=seed) # from Layer.__init__
        self.keep_rate = 1 - rate

        stream_id = Dropout._stream_counter
        Dropout._stream_counter += 1
        self.seed = self._derive_stream_seed(base_seed=self.seed, stream_id=stream_id)
        self._call_counter = 0 # bumped once per training-mode forward call

        self.forward = self._forward_fallback
        self.backward = self._backward_fallback

    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to map low-level hardware paths."""
        if device == 'cupy' and gpu_dropout.is_gpu_dropout_available():
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

        self.output = gpu_dropout.philox_dropout_forward(
            inputs, self.seed, offset, float(self.keep_rate)
        ) # type: ignore
        return self.output

    def _backward_gpu(self, dvalues): 

        self.dinputs = gpu_dropout.philox_dropout_backward(
            dvalues, self.seed, self.offset, float(self.keep_rate)
        ) # type: ignore
        return self.dinputs
    
    def _forward_fallback(self, inputs, training):

        if not training:
            self.output = inputs.copy()
            return self.output
        
        xp = config.get_array_module(inputs)
        self.binary_mask = xp.random.binomial(1, self.keep_rate, size=inputs.shape) \
                            / self.keep_rate
        self.output = inputs * self.binary_mask
        return self.output

    def _backward_fallback(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
        return self.dinputs

class SpatialDropout(Layer): 
    _stream_counter = 0

    def __init__(self, rate, seed=None):

        super().__init__(seed=seed) # from Layer.__init__
        self.rate = rate
        self.keep_rate = 1 - rate

        stream_id = SpatialDropout._stream_counter
        SpatialDropout._stream_counter += 1
        self.seed = self._derive_stream_seed(base_seed=self.seed, stream_id=stream_id) 
        self._call_counter = 0 # bumped once per training-mode forward call

        self.forward = self._forward_fallback
        self.backward = self._backward_fallback

    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to map low-level hardware paths."""
        if device == 'cupy' and gpu_dropout.is_gpu_spatial_dropout_available():
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

        # Stash the channel count such that each backward pass 
        # can rebuild the philox PRNG mask
        self.C = inputs.shape[-1]

        self.output = gpu_dropout.philox_spatial_dropout_forward(
            inputs, self.seed, offset, float(self.keep_rate), self.C
        ) # type: ignore
        return self.output

    def _backward_gpu(self, dvalues):
        self.dinputs = gpu_dropout.philox_spatial_dropout_backward(
            dvalues, self.seed, self.offset, float(self.keep_rate), self.C
        ) # type: ignore
        return self.dinputs
    
    def _forward_fallback(self, inputs, training):
        
        xp = config.get_array_module(inputs)
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return self.output
        
        C = self.inputs.shape[-1]
        self.channel_mask = xp.random.binomial(1, self.keep_rate, size = (1, 1, 1, C)) \
                            / self.keep_rate
        self.output = inputs * self.channel_mask

        return self.output
    
    def _backward_fallback(self, dvalues): 
        self.dinputs = dvalues * self.channel_mask
        return self.dinputs