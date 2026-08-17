import numpy as np
import aether.config as config
from aether.base import Layer
from aether.custom_kernels import dropout_kernel as gpu_dropout

class DropoutRNGState:

    _stream_counter = 0

    def __init__(self, base_seed=None):
        self.stream_id = DropoutRNGState._stream_counter
        DropoutRNGState._stream_counter += 1

        self.seed = self._derive_stream_seed(base_seed, self.stream_id)
        self.offset = 0
    def step(self):
        """Advances the offset for a training step by 1"""
        self.offset += 1
        return self.offset
    
    @staticmethod
    def _derive_stream_seed(base_seed, stream_id):
        """
        Derives a deterministic 64-bit seed for a sepcific random stream. 
        Passing in the same `base_seed` and `stream_id` guarantees identical output
        across runs. 
        """
        if base_seed is None: 
            entropy = None
            spawn_key = (int(stream_id),)
        else:
            entropy = [int(base_seed), int(stream_id)]
            spawn_key = ()
        
        seed_seq = np.random.SeedSequence(entropy, spawn_key=spawn_key)
        return int(seed_seq.generate_state(1, dtype=np.uint64)[0])
    
class Dropout(Layer): 
    def __init__(self, rate, seed=None): 
        self.keep_rate = 1 - rate

        self.rng = DropoutRNGState(base_seed=seed)
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

        offset = self.rng.step()

        self.output = gpu_dropout.philox_dropout_forward(
            inputs, self.rng.seed, offset, float(self.keep_rate)
        ) # type: ignore
        return self.output

    def _backward_gpu(self, dvalues): 

        self.dinputs = gpu_dropout.philox_dropout_backward(
            dvalues, self.rng.seed, self.rng.offset, float(self.keep_rate)
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
    
    def __init__(self, rate, seed=None):

        self.rate = rate
        self.keep_rate = 1 - rate

        self.rng = DropoutRNGState(base_seed=seed)

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

        offset = self.rng.step()

        # Stash the channel count such that each backward pass 
        # can rebuild the philox PRNG mask
        self.C = inputs.shape[-1]

        self.output = gpu_dropout.philox_spatial_dropout_forward(
            inputs, self.rng.seed, offset, float(self.keep_rate), self.C
        ) # type: ignore
        return self.output

    def _backward_gpu(self, dvalues):
        self.dinputs = gpu_dropout.philox_spatial_dropout_backward(
            dvalues, self.rng.seed, self.rng.offset, float(self.keep_rate), self.C
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