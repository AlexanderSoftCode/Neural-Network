import aether.config as config 
from tests.base_case import AetherBaseLayerTestCase
from aether.layers.dropout import Dropout

TARGET_LAYER = Dropout

backends_to_test = ['numpy']
try:
    import cupy as cp
    backends_to_test.append('cupy')
except (ImportError, ModuleNotFoundError):
    pass

def make_suite(backend_name, Layer_Class):
    class_name = f"Test_{Layer_Class.__name__}_{backend_name.upper()}"
    class TestDropout(AetherBaseLayerTestCase):
        DEFAULT_RATE = 0.5
        FIXED_SEED = 12345
        def setUp(self):
            self.backend_name = backend_name
            config.set_backend(backend_name=self.backend_name)
            self.xp = config.xp 

            self.layer = self.make_layer(
                Layer_Class, rate=self.DEFAULT_RATE, seed=self.FIXED_SEED
                )
            self.uses_gpu_kernel = (self.layer.forward.__name__ == '_forward_gpu')
         
        def test_keep_rate_computed_from_rate(self):
            rate = 0.35
            layer = self.make_layer(Layer_Class, rate=rate)
            self.assertAlmostEqual(layer.keep_rate, 1 - rate, places=7)
  
        def test_forward_output_shape_matches_input_1d(self):
            inputs = self.xp.ones(1000, dtype=self.xp.float32)
            output = self.layer.forward(inputs, training=True)
            self.assertEqual(output.shape, inputs.shape)
 
        def test_forward_preserves_multidimensional_input_shape(self):
            """Dropout is an elementwise layer, so it should work on inputs
            of any rank (e.g. (batch, features) activations coming out of a
            Dense/Conv layer), not just flat vectors."""
            inputs = self.xp.ones((8, 16), dtype=self.xp.float32)
            output = self.layer.forward(inputs, training=True)
            self.assertEqual(output.shape, inputs.shape)
 
        def test_forward_eval_mode_is_identity_copy(self):
            inputs = self.xp.arange(20, dtype=self.xp.float32).reshape(4, 5)
            output = self.layer.forward(inputs, training=False)
            self.assertTrue(bool(self.xp.all(output == inputs)))
 
            # Confirm it's a copy, not a shared buffer.
            output[0, 0] = 999
            self.assertFalse(bool(inputs[0, 0] == 999))
  
        def test_forward_training_zero_rate_is_identity(self):
            layer = self.make_layer(Layer_Class, rate=0.0)
            inputs = self.xp.linspace(0.1, 5.0, 500, dtype=self.xp.float32)
            output = layer.forward(inputs, training=True)
            self.assertTrue(bool(self.xp.allclose(output, inputs, rtol=1e-5)))
 
        def test_forward_scales_kept_units_by_inverse_keep_rate(self):
            rate = 0.3
            keep_rate = 1 - rate
            layer = self.make_layer(Layer_Class, rate=rate)
            n = 20000
            inputs = self.xp.ones(n, dtype=self.xp.float32)
 
            if not self.uses_gpu_kernel:
                self.xp.random.seed(self.FIXED_SEED)
 
            output = layer.forward(inputs, training=True)
            kept = output[output != 0]
            self.assertTrue(kept.size > 0)
            self.assertTrue(bool(self.xp.allclose(kept, 1.0 / keep_rate, rtol=1e-4)))
 
        def test_forward_drops_approximately_expected_fraction(self):
            rate = 0.4
            layer = self.make_layer(Layer_Class, rate=rate)
            n = 200000
            inputs = self.xp.ones(n, dtype=self.xp.float32)
 
            if not self.uses_gpu_kernel:
                self.xp.random.seed(self.FIXED_SEED)
 
            output = layer.forward(inputs, training=True)
            dropped_fraction = float(self.xp.mean((output == 0).astype(self.xp.float32)))
            self.assertAlmostEqual(dropped_fraction, rate, delta=0.02)
  
        def test_backward_uses_same_mask_and_scale_as_forward(self):
            # With inputs == dvalues == 1, forward's kept-unit value
            # (x / keep_prob) and backward's kept-unit value
            # (dvalue / keep_prob) land on exactly the same numbers *and*
            # the same positions only if forward/backward share one mask
            # (i.e. backward correctly reuses the offset/mask from the
            # preceding forward call rather than drawing a fresh one).
            layer = self.make_layer(Layer_Class, rate=0.5)
            n = 5000
            inputs = self.xp.ones(n, dtype=self.xp.float32)
 
            forward_out = layer.forward(inputs, training=True)
            dvalues = self.xp.ones(n, dtype=self.xp.float32)
            dinputs = layer.backward(dvalues)
 
            self.assertTrue(bool(self.xp.allclose(forward_out, dinputs)))
 
        def test_repeated_forward_calls_use_different_masks(self):
            n = 5000
            inputs = self.xp.ones(n, dtype=self.xp.float32)
            first = self.layer.forward(inputs, training=True)
            second = self.layer.forward(inputs, training=True)
            self.assertFalse(bool(self.xp.all(first == second)))
 
        # ---- philox / GPU-path bookkeeping --------------------------------
 
        def test_call_counter_increments_on_gpu_path(self):
            if not self.uses_gpu_kernel:
                self.skipTest('_call_counter/offset bookkeeping only applies to the philox GPU path')
 
            layer = self.make_layer(Layer_Class, rate=0.5)
            inputs = self.xp.ones(100, dtype=self.xp.float32)
 
            self.assertEqual(layer._call_counter, 0)
            layer.forward(inputs, training=True)
            self.assertEqual(layer._call_counter, 1)
            self.assertEqual(layer.offset, 1)
            layer.forward(inputs, training=True)
            self.assertEqual(layer._call_counter, 2)
            self.assertEqual(layer.offset, 2)
 
        def test_eval_mode_does_not_bump_call_counter(self):
            if not self.uses_gpu_kernel:
                self.skipTest('_call_counter bookkeeping only applies to the philox GPU path')
 
            layer = self.make_layer(Layer_Class, rate=0.5)
            inputs = self.xp.ones(100, dtype=self.xp.float32)
            layer.forward(inputs, training=False)
            self.assertEqual(layer._call_counter, 0)
 
    TestDropout.__name__ = class_name
    TestDropout.__qualname__ = class_name

    return TestDropout

for backend in backends_to_test:

    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"

    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)
