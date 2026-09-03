import aether.config as config
import tests.base_case as base_case

from aether.layers.dropout import Dropout


class TestDropout(base_case.AetherBaseLayerTestCase):
    DEFAULT_RATE = 0.5
    FIXED_SEED = 12345

    def setUp(self):
        super().setUp()

        self.layer = self.make_built_layer(
            Dropout, input_shape=(16,), rate=self.DEFAULT_RATE, seed=self.FIXED_SEED
        )
        self.uses_gpu_kernel = (self.layer.forward.__name__ == '_forward_gpu')
     
    def test_keep_rate_computed_from_rate(self):
        rate = 0.35
        layer = self.make_built_layer(Dropout, input_shape=(16,), rate=rate)
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

    def test_forward_eval_mode_returns_input_unchanged(self):
        inputs = self.xp.arange(20, dtype=self.xp.float32).reshape(4, 5)
        output = self.layer.forward(inputs, training=False)
        self.assertTrue(bool(self.xp.all(output == inputs)))


    def test_forward_training_zero_rate_is_identity(self):
        layer = self.make_built_layer(Dropout, input_shape=(500,), rate=0.0)
        inputs = self.xp.linspace(0.1, 5.0, 500, dtype=self.xp.float32)
        output = layer.forward(inputs, training=True)
        self.assertTrue(bool(self.xp.allclose(output, inputs, rtol=1e-5)))

    def test_forward_scales_kept_units_by_inverse_keep_rate(self):
        rate = 0.3
        keep_rate = 1 - rate
        n = 20000
        layer = self.make_built_layer(Dropout, input_shape=(n,), rate=rate)
        
        inputs = self.xp.ones(n, dtype=self.xp.float32)

        if not self.uses_gpu_kernel:
            self.xp.random.seed(self.FIXED_SEED)

        output = layer.forward(inputs, training=True)
        kept = output[output != 0]
        self.assertTrue(kept.size > 0)
        self.assertTrue(bool(self.xp.allclose(kept, 1.0 / keep_rate, rtol=1e-4)))

    def test_forward_drops_approximately_expected_fraction(self):
        rate = 0.4
        n = 100000
        layer = self.make_built_layer(Dropout, input_shape=(n,), rate=rate)

        inputs = self.xp.ones(n, dtype=self.xp.float32)

        if not self.uses_gpu_kernel:
            self.xp.random.seed(self.FIXED_SEED)

        output = layer.forward(inputs, training=True)
        dropped_fraction = float(self.xp.mean((output == 0).astype(self.xp.float32)))
        self.assertAlmostEqual(dropped_fraction, rate, delta=0.02)

    def test_backward_uses_same_mask_and_scale_as_forward(self):
        n = 5000
        layer = self.make_built_layer(Dropout, input_shape=(n,), rate=0.5)
        
        inputs = self.xp.ones(n, dtype=self.xp.float32)

        forward_out = layer.forward(inputs, training=True)
        dvalues = self.xp.ones(n, dtype=self.xp.float32)
        dinputs = layer.backward(dvalues)

        self.assertTrue(bool(self.xp.allclose(forward_out, dinputs)))

    def test_repeated_forward_in_same_step_reuses_mask(self):
        n = 50
        inputs = self.xp.ones(n, dtype=self.xp.float32)
        first = self.layer.forward(inputs, training=True).copy()
        second = self.layer.forward(inputs, training=True)
        self.assertTrue(bool(self.xp.all(first == second)))

    # ---- clock / offset bookkeeping -----------------------------------
    # These reach into _clock and _active_offset deliberately: the contract under
    # test *is* the RNG bookkeeping, so the coupling to internals is intentional.

    def test_forward_does_not_advance_the_clock(self):
        layer = self.make_built_layer(Dropout, input_shape=(100,), rate=0.5)
        inputs = self.xp.ones(100, dtype=self.xp.float32)

        # The layer reads the clock but never steps it; the training loop owns advancement.
        self.assertEqual(layer._clock.value, 0)
        layer.forward(inputs, training=True)
        self.assertEqual(layer._clock.value, 0)
        layer.forward(inputs, training=True)
        self.assertEqual(layer._clock.value, 0)

    def test_active_offset_tracks_the_clock(self):
        layer = self.make_built_layer(Dropout, input_shape=(100,), rate=0.5)
        inputs = self.xp.ones(100, dtype=self.xp.float32)

        layer.forward(inputs, training=True)
        self.assertEqual(layer._active_offset, 0)

        layer._clock.advance()
        layer.forward(inputs, training=True)
        self.assertEqual(layer._active_offset, 1)

    def test_eval_mode_clears_active_offset(self):
        layer = self.make_built_layer(Dropout, input_shape=(100,), rate=0.5)
        inputs = self.xp.ones(100, dtype=self.xp.float32)

        # Run a training pass first so the reset is actually observable.
        layer.forward(inputs, training=True)
        layer.forward(inputs, training=False)

        self.assertEqual(layer._clock.value, 0)
        self.assertEqual(layer._active_offset, -1)

base_case.register_test_suites(globals(), TestDropout)