import numpy as np 
import aether.config as config

class Layer:
    precision_policy = None
    def __init__(self, seed: int | None = None):
        self.seed = seed

    def _apply_precision(self, policy):
        self.precision_policy = policy

    def _resolve_compute_dtype(self, xp):
        """
        Resolves against a live xp handle the caller already has
        EX: `xp = config.get_array_module(inputs)` inside forward/backward
        Layer deliberately never caches its own xp refernce
        """

        name = self.precision_policy.compute_dtype_name if self.precision_policy else 'float32'
        return xp.dtype(name)
    
    def _compile_for_device(self, device):
        """
        Hook for layers utilizing fused modules to 
        re-bind runtime pointers
        """
        pass

    def to(self, device):
        """
        Polymorphic state migration engine.
        Each layer isolates its own hardware allocations.
        """
        # Trigger compile/pointer swaps if specific layer overrides it
        self._compile_for_device(device)

        # Locate and migrate any tracking arrays/tensors
        # This catches, weights, biases, or future states dynamically
        for attr_name, attr_value in self.__dict__.items():
            # If attribute is a tensor, shift its VRAM boundary safely
            if hasattr(attr_value, 'shape'):
                setattr(self, attr_name, config.to_device(attr_value, target=device))
