import aether.config as config
from tests.base_case import AetherBaseLayerTestCase

from aether.layers.dropout import SpatialDropout
TARGET_LAYER = SpatialDropout

backends_to_test = ['numpy']

try:
    import cupy as cp
    backends_to_test.append('cupy')

except (ImportError, Exception):
    pass  

def make_suite(backend_name, Layer_Class):
    class_name = f"Test_{Layer_Class.__name__}_{backend_name.upper()}"

    class TestSpatialDropout(AetherBaseLayerTestCase):
        DEFAULT_RATE = 0.5
        FIXED_SEED = 12345
        def setUp(self):
            config.set_backend(backend_name=backend_name)
            self.xp = config.xp
            self.backend_name = backend_name

            self.layer = self.make_layer(
                Layer_Class, rate=self.DEFAULT_RATE, seed=self.FIXED_SEED
            )
            # Skip certain GPU tests using the flag when running on np backend
            self.uses_gpu_kernel = (self.layer.forward.__name__ == '_forward_gpu')

        def test_keep_rate_computed_from_rate(self):
            rate = 0.35
            layer = self.make_layer(Layer_Class, rate=rate)
            self.assertAlmostEqual(layer.keep_rate, 1-rate, places=7)
        # Implement more tests if needed by creating more functions 
        
        def test_forward_output_shape_matches_input_Id(self):
            inputs = self.xp.ones((2, 28, 28, 8), dtype=self.xp.float32)
            output = self.layer.forward(inputs, training=True)

            self.assertEqual(output.shape, inputs.shape)

        def test_forward_eval_mode_is_identity_copy(self):
            # 2 * 4 * 4 * 3 = 96 total elements
            inputs = self.xp.arange(96, dtype=self.xp.float32).reshape(2,4,4,3)
            output = self.layer.forward(inputs, training=False)

            self.assertTrue(bool(self.xp.all(output == inputs)))

            # Assert that its a defensive copy
            output[0, 0, 0, 0] = 999.0
            self.assertNotEqual(inputs[0, 0, 0, 0].item(), 999.0)
        def test_forward_training_zero_rate_is_identity(self):
            layer = self.make_layer(Layer_Class, rate=0.0)
            # 2 * 10 * 10 * 5 = 1000 total elements
            inputs = self.xp.linspace(0.1, 5.0, 1000, dtype=self.xp.float32).reshape(2, 10, 10, 5)
            output = layer.forward(inputs, training=True)
            self.xp.testing.assert_allclose(inputs, output, rtol=1e-4)

        def test_forward_scales_kept_units_by_inverse_keep_rate(self):
            rate = 0.3
            keep_rate = 1 - rate
            layer = self.make_layer(Layer_Class, rate=rate)
            tensor_shape = (3, 10, 10, 6)
            inputs = self.xp.ones(shape=tensor_shape, dtype=self.xp.float32)

            if not self.uses_gpu_kernel:
                self.xp.random.seed(self.FIXED_SEED)

            output = layer.forward(inputs, training=True)
            kept = output[output!=0]
            self.assertTrue(kept.size > 0)
            self.assertTrue(bool(self.xp.allclose(kept, 1.0 / keep_rate, rtol = 1e-4)))

        def test_forward_drops_approximately_expected_fraction(self):
            rate = 0.4
            layer = self.make_layer(Layer_Class, rate=rate)
            tensor_shape = (20, 10, 10, 100)
            inputs = self.xp.ones(shape=tensor_shape, dtype=self.xp.float32)

            if not self.uses_gpu_kernel:
                self.xp.random.seed(self.FIXED_SEED)

            output = layer.forward(inputs, training=True)
            dropped_fraction = float(self.xp.mean((output == 0).astype(self.xp.float32)))
            self.assertAlmostEqual(dropped_fraction, rate, delta=0.08)

        def test_backward_uses_same_mask_and_scale_as_forward(self):
            layer = self.make_layer(Layer_Class, rate=0.5)
            tensor_shape = (1, 4, 4, 20)
            inputs = self.xp.ones(shape=tensor_shape, dtype=self.xp.float32)
            forward_out = layer.forward(inputs, training=True)
            dvalues = self.xp.ones(tensor_shape, dtype=self.xp.float32)
            dinputs = layer.backward(dvalues)

            self.assertTrue(bool(self.xp.allclose(forward_out, dinputs)))

        def test_repeated_forward_calls_use_different_masks(self): 
            tensor_shape = (2, 10, 10, 25)
            inputs = self.xp.ones(tensor_shape, dtype=self.xp.float32)
            first = self.layer.forward(inputs, training=True)
            second = self.layer.forward(inputs, training=True)
            self.assertFalse(bool(self.xp.all(first == second)))

        # ---- philox / GPU-path bookkeeping --------------------------------

        def test_call_counter_increments_on_gpu_path(self):
            if not self.uses_gpu_kernel:
                self.skipTest('offset bookkeeping only applies to the philox GPU path')

            layer = self.make_layer(Layer_Class, rate=0.5)
            tensor_shape = (2, 5, 5, 25)
            inputs = self.xp.ones(shape=tensor_shape, dtype=self.xp.float32)

            self.assertEqual(layer.rng.offset, 0)
            layer.forward(inputs, training=True)
            self.assertEqual(layer.rng.offset, 1)
            layer.forward(inputs, training=True)
            self.assertEqual(layer.rng.offset, 2)

        def test_eval_mode_does_not_bump_call_counter(self):
            if not self.uses_gpu_kernel:
                self.skipTest('offset bookkeeping only applies to the philox GPU path')

            tensor_shape = (2, 5, 5, 25)
            layer = self.make_layer(Layer_Class, rate=0.5)
            inputs = self.xp.ones(shape=tensor_shape, dtype=self.xp.float32)
            layer.forward(inputs, training=False)
            self.assertEqual(layer.rng.offset, 0)

    TestSpatialDropout.__name__ = class_name
    TestSpatialDropout.__qualname__ = class_name

    return TestSpatialDropout

for backend in backends_to_test:
    
    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"

    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)