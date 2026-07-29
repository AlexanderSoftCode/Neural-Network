import unittest
import numpy as np
from aether.config import set_backend
import aether.config as config

try:
    import cupy as cp
except ImportError:
    cp = None

class AetherBaseTestCase(unittest.TestCase):
    """
    A foundational test case structure providing global array module
    pointers and standard environment cleaning methods between hardware
    backend swaps.
    """

    def make_layer(self, layer_cls, **kwargs):
        """Construct a layer and, if it supports device compilation,
        bind it to the currently active backend.
        Requires self.backend_name to be set in setUp before use.
        """
        layer = layer_cls(**kwargs)
        if hasattr(layer, '_compile_for_device'):
            layer._compile_for_device(self.backend_name)
        return layer
    
    def tearDown(self):
        """Reset tracking state to system NumPy default safely between tests."""
        if config.HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        set_backend(backend_name='numpy')