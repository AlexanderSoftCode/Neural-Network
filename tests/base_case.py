import warnings
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

    def make_built_layer(self, layer_cls, input_shape: tuple[int, ...], seed: int | None = None, **kwargs):
        """Constructs, binds to the active device backend, and builds a layer instance.

        This test helper standardizes the parameterized layer lifecycle by first
        instantiating the layer, triggering device compilation/runtime pointer 
        rebinding if supported, and finally executing the layer's build routine to 
        allocate parameter buffers and compute output spatial shapes.

        Args:
            layer_cls: The layer class to instantiate (e.g., `Conv`, `Dense`).
            input_shape: Feature/spatial shape tuple EXCLUDING the batch dimension
                (e.g., `(28, 28, 1)` for 2D Conv or `(128,)` for Dense).
            seed: Optional integer seed for deterministic parameter initialization.
                Defaults to None.
            **kwargs: Arbitrary keyword arguments forwarded directly to the
                `layer_cls` constructor (e.g., `out_channels`, `stride`, `padding`).

        Returns:
            Layer: A fully initialized, device-compiled, and built layer instance.

        Example:
            >>> class TestConvLayer(AetherBaseLayerTestCase):
            ...     def setUp(self):
            ...         super().setUp()
            ...         self.layer = self.make_built_layer(
            ...             Conv,
            ...             input_shape=(28, 28, 1),
            ...             seed=42,
            ...             in_channels=1,
            ...             out_channels=16,
            ...             filter_size=(3, 3),
            ...             stride=(1, 1),
            ...             padding="same",
            ...         )
        """
        layer = layer_cls(**kwargs)

        if hasattr(layer, "_compile_for_device"):
            layer._compile_for_device(self.backend_name)

        layer.build(input_shape, seed=seed)
        return layer
    def make_component(self, component_cls, **kwargs):
        """Instantiates any non-layer aether-component (e.g. Loss, Optimizer, etc.)
        and compiles backend_specific-kernels if supported.
        """
        instance = component_cls(**kwargs)
        if hasattr(instance, "_compile_for_device"):
            instance._compile_for_device(self.backend_name)
        return instance
    def set_precision(self, layer, compute_dtype):
        """Helper mirroring Model.set_precision behavior for an individual layer."""
        # Since I find them annoying, we'll ignore the known emulation warnings after 
        # test setup
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*NumPy float16 is emulated.*",
                category=UserWarning
            )
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