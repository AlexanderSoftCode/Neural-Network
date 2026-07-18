import unittest
import numpy as np 

from aether.config import set_backend
import aether.config as config
from tests.base_case import AetherBaseTestCase

from aether.blocks.activations import ReLU
TARGET_LAYER = ReLU

backends_to_test = ['numpy']
try:
    import cupy as cp
    backends_to_test.append('cupy')

except (ImportError, Exception):
    pass  

def make_suite(backend_name, Layer_Class):

    class_name = f"Test_{Layer_Class.__name__}_{backend_name.upper()}"
    class TestReLU(AetherBaseTestCase):
        def setUp(self):
            set_backend(backend_name=backend_name)
            self.xp = config.xp
            
            self.layer = Layer_Class()

            if hasattr(self.layer, '_compile_for_device'):
                self.layer._compile_for_device(backend_name)

        def test_forward_pass(self):
            """Verify that the layer correctly outputs max(0, x)."""
            
            # Setup input with negative, zero, and positive values
            inputs = self.xp.array([
                [-3.0, -1.0, 0.0], 
                [0.5, 2.0, 5.0]
            ], dtype=self.xp.float32)
            
            expected_output = self.xp.array([
                [0.0, 0.0, 0.0], 
                [0.5, 2.0, 5.0]
            ], dtype=self.xp.float32)
            
            # Execute forward pass using the instance layer
            self.layer.forward(inputs, training=False)
            
            actual_output = self.layer.output
            
            self.xp.testing.assert_array_almost_equal(
                actual_output, 
                expected_output, 
                decimal=4,
                err_msg="Forward pass failed: did not clamp negative values correctly."
            )

        def test_backward_pass(self):
            """Verify that gradients only flow through positive input paths."""
            
            # Inputs to dictate the active/inactive mask
            inputs = self.xp.array([
                [-2.0, 0.0, 1.0, 3.0]
            ], dtype=self.xp.float32)
            
            # Upstream gradients received from the next layer
            dvalues = self.xp.array([
                [0.5, 0.5, 0.5, 0.5]
            ], dtype=self.xp.float32)
            
            # Expected: 0 gradient for inputs <= 0, and passed-through gradient for inputs > 0
            expected_dinputs = self.xp.array([
                [0.0, 0.0, 0.5, 0.5]
            ], dtype=self.xp.float32)
            
            self.layer.forward(inputs, training=False)
            self.layer.backward(dvalues)
            
            actual_dinputs = self.layer.dinputs
            
            self.xp.testing.assert_array_almost_equal(
                actual_dinputs, 
                expected_dinputs, 
                decimal=4,
                err_msg="Backward pass failed: gradient routing mismatch."
            )
    TestReLU.__name__ = class_name
    TestReLU.__qualname__ = class_name
            
    return TestReLU

for backend in backends_to_test:
    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"

    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)