import aether.config as config
import tests.base_case as base_case

from aether.layers.normalization import BatchNorm


class TestBatchNormLayer(base_case.AetherBaseLayerTestCase):
    NUM_FEATURES_DENSE = 16
    DENSE_SHAPE = (8, 16)               

    NUM_CHANNELS_CONV = 8
    CONV_SHAPE = (4, 14, 14, 8)

    EPSILON = 1e-5
    MOMENTUM = 0.9

    def setUp(self):
        super().setUp()

        # 2D Dense BatchNorm instance
        self.layer_dense = self.make_built_layer(
            BatchNorm, 
            input_shape=(self.DENSE_SHAPE[1],),
            epsilon=self.EPSILON, 
            momentum=self.MOMENTUM, 
        )

        # 4D Conv BatchNorm instance
        self.layer_conv = self.make_built_layer(
            BatchNorm, 
            input_shape=self.CONV_SHAPE[1:],
            epsilon=self.EPSILON, 
            momentum=self.MOMENTUM, 
        )

        # Seeded input batches
        self.xp.random.seed(42)
        self.dense_inputs = (
            self.xp.random.randn(*self.DENSE_SHAPE).astype(self.xp.float32) * 5.0 + 3.0
        )
        self.conv_inputs = (
            self.xp.random.randn(*self.CONV_SHAPE).astype(self.xp.float32) * 3.0 - 2.0
        )

    def test_build_parameter_initialization(self):
        """Verify gamma, beta, running mean, and running variance allocation."""
        # Trainable parameters
        self.assertEqual(self.layer_dense.gamma.shape, (self.NUM_FEATURES_DENSE,))
        self.assertEqual(self.layer_dense.beta.shape, (self.NUM_FEATURES_DENSE,))
        self.assertTrue(self.xp.allclose(self.layer_dense.gamma, 1.0))
        self.assertTrue(self.xp.allclose(self.layer_dense.beta, 0.0))

        # Optimizer aliases
        self.assertIs(self.layer_dense.weights, self.layer_dense.gamma)
        self.assertIs(self.layer_dense.biases, self.layer_dense.beta)

        # Non-trainable running statistics
        self.assertEqual(self.layer_dense.running_mean.shape, (self.NUM_FEATURES_DENSE,))
        self.assertEqual(self.layer_dense.running_var.shape, (self.NUM_FEATURES_DENSE,))
        self.assertTrue(self.xp.allclose(self.layer_dense.running_mean, 0.0))
        self.assertTrue(self.xp.allclose(self.layer_dense.running_var, 1.0))

    def test_forward_training_normalization_2d(self):
        """Verify that 2D batch outputs are normalized to zero mean and unit variance."""
        out = self.layer_dense.forward(self.dense_inputs, training=True)

        self.assertEqual(out.shape, self.DENSE_SHAPE)
        
        # Batch mean across batch dimension should be close to 0 (beta=0)
        batch_mean = self.xp.mean(out, axis=0)
        self.assertTrue(self.xp.allclose(batch_mean, 0.0, atol=1e-5))

        # Batch variance across batch dimension should be close to 1 (gamma=1)
        batch_var = self.xp.var(out, axis=0)
        self.assertTrue(self.xp.allclose(batch_var, 1.0, atol=1e-3))

    def test_forward_training_normalization_4d(self):
        """Verify that 4D CNN outputs are normalized across spatial and batch axes."""
        out = self.layer_conv.forward(self.conv_inputs, training=True)

        self.assertEqual(out.shape, self.CONV_SHAPE)
        
        batch_mean = self.xp.mean(out, axis=(0, 1, 2))
        batch_var = self.xp.var(out, axis=(0, 1, 2))

        self.assertTrue(self.xp.allclose(batch_mean, 0.0, atol=1e-4))
        self.assertTrue(self.xp.allclose(batch_var, 1.0, atol=1e-3))

    def test_running_statistics_ema_update(self):
        """Verify the exponential moving average update for running statistics."""
        initial_mean = self.layer_dense.running_mean.copy()
        initial_var = self.layer_dense.running_var.copy()

        batch_mean = self.xp.mean(self.dense_inputs, axis=0)
        batch_var = self.xp.var(self.dense_inputs, axis=0)

        self.layer_dense.forward(self.dense_inputs, training=True)

        expected_running_mean = (
            self.MOMENTUM * initial_mean + (1.0 - self.MOMENTUM) * batch_mean
        )
        expected_running_var = (
            self.MOMENTUM * initial_var + (1.0 - self.MOMENTUM) * batch_var
        )

        self.assertTrue(self.xp.allclose(self.layer_dense.running_mean, expected_running_mean))
        self.assertTrue(self.xp.allclose(self.layer_dense.running_var, expected_running_var))

    def test_forward_inference_mode(self):
        """Verify inference mode uses running stats and does not mutate them or store caches."""
        # Populate running statistics with custom values
        test_mean = self.xp.full((self.NUM_FEATURES_DENSE,), 3.0, dtype=self.xp.float32)
        test_var = self.xp.full((self.NUM_FEATURES_DENSE,), 4.0, dtype=self.xp.float32)
        self.layer_dense.running_mean = test_mean.copy()
        self.layer_dense.running_var = test_var.copy()

        out = self.layer_dense.forward(self.dense_inputs, training=False)

        # Check mathematical result: (x - test_mean) / sqrt(test_var + eps)
        expected = (self.dense_inputs - 3.0) / self.xp.sqrt(4.0 + self.EPSILON)
        self.assertTrue(self.xp.allclose(out, expected, atol=1e-5))

        # Running statistics must remain untouched
        self.assertTrue(self.xp.allclose(self.layer_dense.running_mean, test_mean))
        self.assertTrue(self.xp.allclose(self.layer_dense.running_var, test_var))

    def test_backward_gradients_shapes_2d(self):
        """Verify gradient shapes for 2D Dense input backpropagation."""
        self.layer_dense.forward(self.dense_inputs, training=True)

        dvalues = self.xp.ones_like(self.dense_inputs)
        dinputs = self.layer_dense.backward(dvalues)

        self.assertEqual(dinputs.shape, self.DENSE_SHAPE)
        self.assertEqual(self.layer_dense.dweights.shape, (self.NUM_FEATURES_DENSE,))
        self.assertEqual(self.layer_dense.dbiases.shape, (self.NUM_FEATURES_DENSE,))

    def test_backward_gradients_shapes_4d(self):
        """Verify gradient shapes for 4D CNN input backpropagation."""
        self.layer_conv.forward(self.conv_inputs, training=True)

        dvalues = self.xp.ones_like(self.conv_inputs)
        dinputs = self.layer_conv.backward(dvalues)

        self.assertEqual(dinputs.shape, self.CONV_SHAPE)
        self.assertEqual(self.layer_conv.dweights.shape, (self.NUM_CHANNELS_CONV,))
        self.assertEqual(self.layer_conv.dbiases.shape, (self.NUM_CHANNELS_CONV,))

    def test_precision_retention_float32(self):
        """Verify calculations do not promote float32 arrays to float64."""
        out = self.layer_conv.forward(self.conv_inputs, training=True)
        dinputs = self.layer_conv.backward(self.xp.ones_like(out))

        self.assertEqual(out.dtype, self.xp.float32)
        self.assertEqual(dinputs.dtype, self.xp.float32)
        self.assertEqual(self.layer_conv.running_mean.dtype, self.xp.float32)
        self.assertEqual(self.layer_conv.running_var.dtype, self.xp.float32)
        self.assertEqual(self.layer_conv.dweights.dtype, self.xp.float32)
        self.assertEqual(self.layer_conv.dbiases.dtype, self.xp.float32)


base_case.register_test_suites(globals(), TestBatchNormLayer)