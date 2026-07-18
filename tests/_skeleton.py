# Template for writing a test file for a layer
import unittest

# Core Framework Configuration Imports
from aether.config import set_backend, get_stride_utility
import aether.config as config
from tests.base_case import AetherBaseTestCase

# Replace this with your specific layout import
from aether.blocks.conv import Conv
TARGET_LAYER = Conv

backends_to_test = ['numpy']
# GPU check & loading loop
try:
    import cupy as cp
    backends_to_test.append('cupy')
    # Assuming the GPU class maps cleanly or is imported from users CuPy file

except (ImportError, Exception):
    pass  # Fall back onto the NumPy suite if CUDA/ROCm isn't present


# Dynamic Factory Class Generation
def make_suite(backend_name, Layer_Class):
    class_name = f"Test_{Layer_Class.__name__}_{backend_name.upper()}"
    # Rewrite `TestLayer` for specific class being tested
    class TestLayer(AetherBaseTestCase):
        def setUp(self):
            # Function from aether.config.py 
            set_backend(backend_name=backend_name)
            self.xp = config.xp

            # Grab extra utility imports for specific backend if needed
            self.as_strided = get_stride_utility(self.xp)
            self.layer = Layer_Class

            # If a backend requires seperate functions, mimic Model.to
            # otherwise do nothing
            if hasattr(self.layer, '_compile_for_device'):
                self.layer._compile_for_device(backend_name)
        
        def test_conv_forward_shape(self):
            """Verify output dimensions based on padding and stride"""
            output = self.layer.forward(self.test_images, training=True)
            # Given padding = 1, kernel = 3, 28x28 remainds 28x28 (same)
            
            expected_shape = (2, 28, 28, 4)
            self.assertEqual(output.shape, expected_shape)

        def test_conv_numerical_gradient_check(self):
            """Verify that analytical backpropagation calculations evaluate properly."""
            pass
        # Implement more tests if needed by creating more functions 
        
    # Metaprogramming class property remapping for clear test-runner outputs
    TestLayer.__name__ = class_name
    TestLayer.__qualname__ = class_name

    return TestLayer

# Global Unpacking Loop
for backend in backends_to_test:
    
    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"

    # Bind generated class to the global namespace for unittest discovery
    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)