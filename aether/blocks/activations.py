import numpy as np
import aether.config as config

class ReLU:

    def forward(self, inputs, training=True):
        xp = config.get_array_module(inputs)
        self.output = xp.maximum(0, inputs)
        return self.output
    
    def backward(self, dvalues): 
        self.dinputs = dvalues * (self.output > 0)
        return self.dinputs


try: 
    import cupy as cp
    @cp.fuse()
    def _fused_leaky_relu_forward(x, alpha):
        return cp.maximum(0, x) + alpha * cp.minimum(0, x)
except (ImportError, Exception):
    _fused_leaky_relu_forward = None

try:
    import cupy as cp
    @cp.fuse()
    def _fused_leaky_relu_backward(dvalues, output, alpha):
        # ROCm/HIP Workaround: Comparing fusion variables against raw Python literals 
        # (e.g., output > 0) causes a type-guessing bug in modern NumPy environments, 
        # throwing an AttributeError on AMD architectures during the AST trace.
        # Generating a zero scalar explicitly typed by multiplying 'alpha * 0'
        # ensures variable-to-variable comparison, bypassing weak type-promotion logic 
        # and compiling cleanly into a branchless SIMD loop on both CUDA and HIP stacks.
        fused_zero = alpha * 0
        return dvalues * (1.0 - (output <= fused_zero) * (1.0 - alpha))
except (ImportError, Exception):
    _fused_leaky_relu_backward = None


class LeakyReLU:

    def __init__(self, alpha=0.01):
        self.alpha = alpha

        self.forward = self._forward_fallback
        self.backward = self._backward_fallback

    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to map low-level hardware paths"""
        if device == 'cupy' and _fused_leaky_relu_forward is not None and _fused_leaky_relu_backward is not None:
            self.forward = self._forward_gpu   
            self.backward = self._backward_gpu
        else:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback

    def _forward_gpu(self, inputs, training=True):
        """Dedicated GPU execution with cp.fuse"""
        self.output = _fused_leaky_relu_forward(inputs, self.alpha)
        return self.output
    
    def _forward_fallback(self, inputs, training=True):
        xp = config.get_array_module(inputs)        
        self.output = xp.maximum(0, inputs) + self.alpha * xp.minimum(0, inputs)
        return self.output
    
    def _backward_gpu(self, dvalues):
        """Dedicated GPU execution with cp.fuse"""
        self.dinputs = _fused_leaky_relu_backward(dvalues, self.output, self.alpha)
        return self.dinputs 
    
    def _backward_fallback(self, dvalues):
        self.dinputs = dvalues * (1.0 - (self.output <= 0) * (1.0 - self.alpha))
        return self.dinputs