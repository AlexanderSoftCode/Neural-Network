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
    __test__ = False  # Suppress standalone discovery of the base class

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        
        if cls.__name__ == 'AetherBaseLayerTestCase':
            cls.__test__ = False
        else:
            cls.__test__ = True

    def shortDescription(self):
        """Docstrings are ommitted when running verbose -v unit tests"""
        return None
    def tearDown(self):
        """Reset tracking state to system NumPy default safely between tests."""
        if config.HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        set_backend(backend_name='numpy')


class AetherBaseLayerTestCase(AetherBaseTestCase):
    """Base class for layers/activations/loss modules"""
    __test__ = False 

    def make_layer(self, layer_cls, **kwargs):
        """Construct a layer and, if it supports device compilation,
        bind it to the currently active backend.
        Requires self.backend_name to be set in setUp before use.
        """
        layer = layer_cls(**kwargs)
        if hasattr(layer, '_compile_for_device'):
            layer._compile_for_device(self.backend_name)
        return layer
    def make_built_layer(self, layer_cls, **kwargs):
        """Construct, bind to backend, and allocate array buffers for 
        parameterized layers (e.g., Dense, Conv2D).
        """
        layer = self.make_layer(layer_cls, **kwargs)
        layer.build()
        return layer

    def set_precision(self, layer, compute_dtype):
        """Helper mirroring Model.set_precision behavior for an individual layer."""
        policy = config.DTypePolicy(compute_dtype=compute_dtype)
        if hasattr(layer, "_apply_precision") and not getattr(
            layer, "_precision_exempt", False
        ):
            layer._apply_precision(policy)
        return policy

    def test_backend_pointer_swap(self):
        """Verify the layer/test environment successfully routes to the correct backend execution hooks."""
        super().setUp()
        if not hasattr(self, 'xp'):
            import numpy as np
            self.xp = np
        if "CUPY" in self.__class__.__name__.upper():
            self.assertEqual(
                self.xp.__name__, 
                "cupy", 
                "Test class marked as CUPY, but self.xp is not Cupy!"
            )
        else:
            self.assertEqual(
                self.xp.__name__, 
                "numpy", 
                "Test class marked for CPU/NumPy, but self.xp is not Numpy!"
            )
