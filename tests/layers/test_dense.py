import numpy as np

from aether.config import set_backend
import aether.config as config
from tests.base_case import AetherBaseLayerTestCase

from aether.layers.linear import Dense
TARGET_LAYER = Dense

backends_to_test = ['numpy']

try:
    import cupy as cp
    backends_to_test.append('cupy')
except:
    pass 

def make_suite(backend_name, Layer_Class):    

    class_name = f"Test_{Layer_Class.__name__}_{backend_name.upper()}"
    class TestDenseLayer(AetherBaseLayerTestCase): 

        N_INPUTS = 8
        N_NEURONS = 4
        BATCH_SIZE = 3
        def setUp(self):
            self.backend_name = backend_name
            set_backend(backend_name=self.backend_name)
            self.xp = config.xp
            self.xp.random.seed(42)

            self.layer = self.make_built_layer(
                Layer_Class, n_inputs=self.N_INPUTS, n_neurons=self.N_NEURONS)
            self.test_inputs = self.xp.random.randn(self.BATCH_SIZE, self.N_INPUTS).astype(self.xp.float32) 

        def test_forward_shape(self):
            "Verify output shape is 10, 3"
            output = self.layer.forward(self.test_inputs, training=True)
            self.assertEqual(output.shape, (self.BATCH_SIZE, self.N_NEURONS))

        def test_parameters_initialization(self):
            """Verify parameter buffer shapes, float32 defaults, and He Initialization"""
            self.assertEqual(self.layer.weights.shape, (self.N_INPUTS, self.N_NEURONS))
            self.assertEqual(self.layer.biases.shape, (1, self.N_NEURONS))
            self.assertEqual(self.layer.weights.dtype, self.xp.float32)
            self.assertEqual(self.layer.biases.dtype, self.xp.float32)

            self.assertTrue(self.xp.all(self.layer.biases == 0))

            # He initialization scale check
            large_layer = self.make_built_layer(Layer_Class, n_inputs=256, n_neurons=256)
            expected_std = float(self.xp.sqrt(2.0 / 256))
            actual_std = float(self.xp.std(large_layer.weights))
            self.assertAlmostEqual(actual_std, expected_std, delta=0.03)


        def test_dense_zero_input(self):
            "Given an input of all zeros, show an output of all zeros (plus biases)"

            layer = self.make_built_layer(Dense, n_inputs = 728, n_neurons=20)
            
            zero_input = self.xp.zeros((1, 728))

            # Every pixel in the output should equal the bias for that layer
            # We reshape the biases to (1, biases) to match output shape

            output = layer.forward(zero_input, training = False)
            expected_output = self.xp.broadcast_to(layer.biases, output.shape)

            # compare the values
            self.xp.testing.assert_array_almost_equal(output, expected_output)

        def test_get_and_set_parameters(self):
            """Verify parameter geter and setter roundtrips."""
            w_orig, b_orig = self.layer.get_parameters()
            new_w = self.xp.ones_like(w_orig) * 0.25
            new_b = self.xp.ones_like(b_orig) * 0.05

            self.layer.set_parameters(new_w, new_b)
            w_res, b_res = self.layer.get_parameters()

            self.assertTrue(self.xp.array_equal(w_res, new_w))
            self.assertTrue(self.xp.array_equal(b_res, new_b))


        def test_forward_training_branch_caching(self):
            """Verify forward(training=True) populates ephemeral compute caches."""
            self.layer.forward(self.test_inputs, training=True)
            self.assertIsNotNone(self.layer.inputs)
            self.assertIsNotNone(self.layer._inputs_compute)
            self.assertIsNotNone(self.layer._weights_compute)

        def test_forward_inference_branch_clears_cache(self):
            """Verify forward(training=False) clears compute caches."""
            self.layer.forward(self.test_inputs, training=True)
            self.layer.forward(self.test_inputs, training=False)

            self.assertIsNone(self.layer.inputs)
            self.assertIsNone(self.layer._inputs_compute)
            self.assertIsNone(self.layer._weights_compute)

        def test_backward_without_training_forward_raises_runtime_error(self):

            dvalues = self.xp.random.randn(self.BATCH_SIZE, self.N_NEURONS).astype(self.xp.float32)

            self.layer.forward(self.test_inputs, training=False)
            with self.assertRaises(RuntimeError):
                self.layer.backward(dvalues)

        def test_backward_gradient_shapes_and_cache_cleanup(self):
            """Ensure backprop computes valid gradient shapes and invalidates cache."""

            output = self.layer.forward(self.test_inputs, training=True)
            dvalues = self.xp.random.randn(*output.shape).astype(self.xp.float32)
            dinputs = self.layer.backward(dvalues)

            self.assertEqual(dinputs.shape, self.test_inputs.shape)
            self.assertEqual(self.layer.dweights.shape, self.layer.weights.shape)
            self.assertEqual(self.layer.dbiases.shape, self.layer.biases.shape)

            # Caches cleared post-backward
            self.assertIsNone(self.layer._inputs_compute)
            self.assertIsNone(self.layer._weights_compute)

        def test_mixed_precision_flow_float16(self):
            """Verify forward compute casting and gradient recovery to float32."""
            self.set_precision(self.layer, compute_dtype='float16')

            output = self.layer.forward(self.test_inputs, training=True)
            self.assertEqual(output.dtype, self.xp.float16)

            dvalues = self.xp.random.randn(*output.shape).astype(self.xp.float32)
            dinputs = self.layer.backward(dvalues)

            # Gradients to previous layer match input precision (float32)
            self.assertEqual(dinputs.dtype, self.test_inputs.dtype)
            # Stored master parameter gradients cast back to param precision (float32)
            self.assertEqual(self.layer.dweights.dtype, self.xp.float32)
            self.assertEqual(self.layer.dbiases.dtype, self.xp.float32)

        def test_numerical_gradient_check(self):
            """Finite difference verification of analytical dweights and dinputs."""
            layer = self.make_built_layer(Layer_Class, n_inputs=4, n_neurons=3)
            layer.weights = layer.weights.astype(self.xp.float64)
            layer.biases = layer.biases.astype(self.xp.float64)

            x = self.xp.random.randn(2, 4).astype(self.xp.float64)
            dvalues = self.xp.random.randn(2, 3).astype(self.xp.float64)
            eps = 1e-6

            # Analytical gradients
            layer.forward(x, training=True)
            layer.backward(dvalues)
            analytical_dweights = layer.dweights.copy()

            # Numerical gradients for weights
            numerical_dweights = self.xp.zeros_like(layer.weights)
            it = np.nditer(config.to_device(layer.weights, 'numpy'), flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index

                layer.weights[idx] += eps
                out_pos = layer.forward(x, training=True)
                loss_pos = self.xp.sum(out_pos * dvalues)

                layer.weights[idx] -= 2 * eps
                out_neg = layer.forward(x, training=True)
                loss_neg = self.xp.sum(out_neg * dvalues)

                layer.weights[idx] += eps  # restore
                numerical_dweights[idx] = (loss_pos - loss_neg) / (2 * eps)
                it.iternext()

            rel_error = self.xp.max(
                self.xp.abs(analytical_dweights - numerical_dweights)
                / (self.xp.abs(analytical_dweights) + self.xp.abs(numerical_dweights) + 1e-8)
            )
            self.assertLess(float(rel_error), 1e-4)

    TestDenseLayer.__name__ = class_name
    TestDenseLayer.__qualname__ = class_name
    return TestDenseLayer 

for backend in backends_to_test:

    class_name = f"Test_{TARGET_LAYER}.__name__)_{backend.upper()}"

    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)