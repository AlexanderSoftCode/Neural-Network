import numpy as np
import aether.config as config
from aether.base import Layer

_philox_dropout_forward = None
_philox_dropout_backward = None 

_PHILOX_PREAMBLE = r'''
__device__ __forceinline__ float philox_uniform(
        unsigned long long philox_seed,
        unsigned long long philox_offset,
        long long idx) {
 
    // 128-bit counter = {offset_lo, offset_hi, idx_lo, idx_hi}
    unsigned int c0 = (unsigned int)(philox_offset & 0xffffffffULL);
    unsigned int c1 = (unsigned int)(philox_offset >> 32);
    unsigned int c2 = (unsigned int)((unsigned long long)idx & 0xffffffffULL);
    unsigned int c3 = (unsigned int)((unsigned long long)idx >> 32);
 
    // 64-bit key = {seed_lo, seed_hi}
    unsigned int k0 = (unsigned int)(philox_seed & 0xffffffffULL);
    unsigned int k1 = (unsigned int)(philox_seed >> 32);
 
    #pragma unroll
    for (int round = 0; round < 10; round++) {
        // 32x32 -> 64 multiply, split into hi/lo. Plain arithmetic (no
        // __umulhi) so this compiles under both NVRTC and HIPRTC.
        unsigned long long p0 = (unsigned long long)0xD2511F53u * c0;
        unsigned long long p1 = (unsigned long long)0xCD9E8D57u * c2;
        unsigned int lo0 = (unsigned int)(p0 & 0xffffffffULL);
        unsigned int hi0 = (unsigned int)(p0 >> 32);
        unsigned int lo1 = (unsigned int)(p1 & 0xffffffffULL);
        unsigned int hi1 = (unsigned int)(p1 >> 32);
 
        unsigned int nc0 = hi1 ^ c1 ^ k0;
        unsigned int nc1 = lo1;
        unsigned int nc2 = hi0 ^ c3 ^ k1;
        unsigned int nc3 = lo0;
 
        c0 = nc0; c1 = nc1; c2 = nc2; c3 = nc3;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return (c0 >> 8) * (1.0f / 16777216.0f);
    // top 24 bits of c0 -> uniform float in [0, 1), same convention
    // curand/PyTorch/TF use for uint32 -> uniform float.
}
'''

_philox_dropout_forward = config.build_kernel(lambda: config.cp.ElementwiseKernel(
    'T x, uint64 philox_seed, uint64 philox_offset, float64 keep_prob',
    'T y',
    '''
    float u = philox_uniform(philox_seed, philox_offset, i);
    y = (u < keep_prob) ? (T)(x / keep_prob) : (T)0;
    ''',
    '_philox_dropout_forward',
    preamble=_PHILOX_PREAMBLE,
    ),
    name = '_philox_dropout_forward'
)

_philox_dropout_backward = config.build_kernel(lambda: config.cp.ElementwiseKernel(
    'T dvalues, uint64 philox_seed, uint64 philox_offset, float64 keep_prob',
    'T dinputs',
    '''
    float u = philox_uniform(philox_seed, philox_offset, i);
    dinputs = (u < keep_prob) ? (T)(dvalues / keep_prob) : (T)0;
    ''',
    'philox_dropout_backward',
    preamble=_PHILOX_PREAMBLE,
    ),
    name = '_philox_dropout_backward'
)
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
            inputs, self.seed, offset, float(self.keep_rate)
        )
        return self.output

    def _backward_gpu(self, dvalues): 

        self.dinputs = _philox_dropout_backward(
            dvalues, self.seed, self.offset, float(self.keep_rate)
        )
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

_philox_spatial_dropout_forward = config.build_kernel(lambda: config.cp.ElementwiseKernel(
    'T x, uint64 philox_seed, uint64 philox_offset, float64 keep_prob, int64 C',
    'T y',
    '''
    // Collapse the per-element index into per-channel index. 
    long long c = i % C;
    float u = philox_uniform(philox_seed, philox_offset, c);
    y = (u < keep_prob) ? (T)(x / keep_prob) : (T)0;
    ''',
    '_philox_spatial_dropout_forward',
    preamble=_PHILOX_PREAMBLE,
    ),
    name = '_philox_spatial_dropout_forward'
)
 
_philox_spatial_dropout_backward = config.build_kernel(lambda: config.cp.ElementwiseKernel(
    'T dvalues, uint64 philox_seed, uint64 philox_offset, float64 keep_prob, int64 C',
    'T dinputs',
    '''
    // Compute the same channel mask used in forward pass, as the output is deterministic
    // passing in our forward inputs results in the same same u value.
    long long c = i % C;
    float u = philox_uniform(philox_seed, philox_offset, c);
    dinputs = (u < keep_prob) ? (T)(dvalues / keep_prob) : (T)0;
    ''',
    '_philox_spatial_dropout_backward',
    preamble=_PHILOX_PREAMBLE,
    ),
    name = '_philox_spatial_dropout_backward'
)
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
        if device == 'cupy' and _philox_spatial_dropout_forward is not None and _philox_spatial_dropout_backward is not None:
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

        self.output = _philox_spatial_dropout_forward(
            inputs, self.seed, offset, float(self.keep_rate), self.C
        )
        return self.output

    def _backward_gpu(self, dvalues):
        self.dinputs = _philox_spatial_dropout_backward(
            dvalues, self.seed, self.offset, float(self.keep_rate), self.C
        )
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