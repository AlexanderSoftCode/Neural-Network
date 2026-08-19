import numpy as np 
import aether.config as config

class Layer:
    _precision_exempt: bool = False
    def __init__(self):
        self.seed = None
        # these attributes below will likely be used for Model.save()
        self.input_shape = None
        self.output_shape = None
        self.weights = None 
        self.biases = None
        self.precision_policy = None

    def _apply_precision(self, policy):
        self.precision_policy = policy
    def build(self, input_shape, seed=None):
        """
        Base build contract: stores shapes and passes through.
        Does not mutate self.seed on stateless layers.
        """
        self.input_shape = input_shape
        self.output_shape = input_shape

        return self.output_shape
    
    def get_config(self) -> dict:
        """Override to return constructor kwargs required to reconstruct the layer."""
        return {}

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

    def get_parameters(self) -> dict[str, object]:
        """Returns a dict of state tensors belonging to this layer."""
        params = {}
        if self.weights is not None:
            params["weights"] = self.weights
        if self.biases is not None:
            params["biases"] = self.biases
        return params

    def set_parameters(self, **params):
        """Sets state tensors to this layer."""
        for name, value in params.items():
            if hasattr(self, name):
                setattr(self, name, value)

"""    def to(self, device):
        
        Polymorphic state migration engine.
        Each layer isolates its own hardware allocations.
        
        # Trigger compile/pointer swaps if specific layer overrides it
        self._compile_for_device(device)

        # Locate and migrate any tracking arrays/tensors
        # This catches, weights, biases, or future states dynamically
        for attr_name, attr_value in self.__dict__.items():
            # If attribute is a tensor, shift its VRAM boundary safely
            if hasattr(attr_value, 'shape'):
                setattr(self, attr_name, config.to_device(attr_value, target=device))
"""