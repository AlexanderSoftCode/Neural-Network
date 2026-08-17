"""
Aether-ML Unit Tests: Flatten Layer
===================================
Tests forward flattening logic, backward gradient reconstruction, 
and batch consistency across NumPy (CPU) and CuPy (GPU) backends.
"""

import aether.config as config
from tests.base_case import AetherBaseLayerTestCase
from aether.layers.linear import Flatten

TARGET_LAYER = Flatten

backends_to_test = ['numpy']
try:
    import cupy as cp
    backends_to_test.append('cupy')
except (ImportError, Exception):
    pass


def make_suite(backend_name, Layer_Class):
    class_name = f"Test_{Layer_Class.__name__}_{backend_name.upper()}"

    class TestFlattenLayer(AetherBaseLayerTestCase):
        INPUT_SHAPE = (4, 7, 7, 3)  # (batch_size, height, width, channels)
        FLATTENED_DIM = 7 * 7 * 3    # 147

        def setUp(self):
            self.backend_name = backend_name
            config.set_backend(backend_name=self.backend_name)
            self.xp = config.xp

            self.layer = self.make_layer(Layer_Class)
            
            self.test_inputs = self.xp.arange(
                4 * 7 * 7 * 3, dtype=self.xp.float32
            ).reshape(self.INPUT_SHAPE)

        def test_flatten_forward_shape(self):
            """Verify output is flattened to 2D (batch_size, total_features)."""
            output = self.layer.forward(self.test_inputs, training=True)
            expected_shape = (self.INPUT_SHAPE[0], self.FLATTENED_DIM)
            
            self.assertEqual(output.shape, expected_shape)
            self.assertTrue(output.flags.c_contiguous)

        def test_flatten_forward_values(self):
            """Verify flattened output maintains contiguous row-major ordering."""
            output = self.layer.forward(self.test_inputs, training=True)
            expected_first_sample = self.test_inputs[0].ravel()
            
            self.assertTrue(
                self.xp.allclose(output[0], expected_first_sample)
            )
            
        def test_flatten_training_caches_shape(self):
            """Verify that inputs_shape IS stored when training=True."""
            self.layer.forward(self.test_inputs, training=True)
            
            self.assertTrue(hasattr(self.layer, 'inputs_shape'))
            self.assertEqual(self.layer.inputs_shape, self.INPUT_SHAPE)

        def test_flatten_inference_does_not_cache_shape(self):
            """Verify that inputs_shape is NOT stored/cached when training=False."""
            self.assertFalse(hasattr(self.layer, 'inputs_shape'))

            output = self.layer.forward(self.test_inputs, training=False)
            
            expected_shape = (self.INPUT_SHAPE[0], self.FLATTENED_DIM)
            self.assertEqual(output.shape, expected_shape)

            self.assertFalse(
                hasattr(self.layer, 'inputs_shape'),
                "Flatten layer should not store `inputs_shape` when training=False."
            )
        def test_flatten_backward_shape(self):
            """Verify backward pass reshapes gradients to match the original input shape."""
            self.layer.forward(self.test_inputs, training=True)
            
            dvalues = self.xp.ones((self.INPUT_SHAPE[0], self.FLATTENED_DIM), dtype=self.xp.float32)
            dinputs = self.layer.backward(dvalues)

            self.assertEqual(dinputs.shape, self.INPUT_SHAPE)

        def test_flatten_gradient_integrity(self):
            """Verify backward pass preserves exact gradient values without distortion."""
            self.layer.forward(self.test_inputs, training=True)
            
            dvalues = self.xp.arange(
                self.INPUT_SHAPE[0] * self.FLATTENED_DIM, dtype=self.xp.float32
            ).reshape(self.INPUT_SHAPE[0], self.FLATTENED_DIM)

            dinputs = self.layer.backward(dvalues)
            
            reconstructed_dvalues = dinputs.reshape(self.INPUT_SHAPE[0], -1)
            self.assertTrue(self.xp.allclose(reconstructed_dvalues, dvalues))

    TestFlattenLayer.__name__ = class_name
    TestFlattenLayer.__qualname__ = class_name
    return TestFlattenLayer


for backend in backends_to_test:
    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"
    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)