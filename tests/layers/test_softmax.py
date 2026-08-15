import aether.config as config
from tests.base_case import AetherBaseLayerTestCase

from aether.layers.activations import SoftMax
TARGET_LAYER = SoftMax

backends_to_test = ['numpy']
try:
    import cupy as cp
    backends_to_test.append('cupy')
except (ImportError, Exception):
    pass

def make_suite(backend_name, Layer_Class):
    class_name = f"Test_{Layer_Class.__name__}_{backend_name.upper()}"

    class TestSoftMax(AetherBaseLayerTestCase):
        def setUp(self):

            self.backend_name = backend_name
            config.set_backend(backend_name=self.backend_name)
            self.xp = config.xp

            self.layer = self.make_layer(Layer_Class)

        def test_forward_probabilities(self):
            """Verify SoftMax bounds values between 0 and 1, and rows sum to 1.0."""
            raw_inputs = self.xp.array([
                [1.0, 2.0, 3.0, 4.0],
                [-10.0, 0.0, 10.0, 5.0]
            ], dtype=self.xp.float32)
            
            output = self.layer.forward(raw_inputs, training=True)
            
            row_sums = self.xp.sum(output, axis=1)
            expected_sums = self.xp.ones(row_sums.shape, dtype=self.xp.float32)
            
            self.xp.testing.assert_array_almost_equal(row_sums, expected_sums, decimal=5)
            self.assertTrue(self.xp.all(output >= 0.0) and self.xp.all(output <= 1.0))

        def test_numerical_stability(self):
            """Verify that extreme logit input scales do not explode or result in NaNs."""
            extreme_inputs = self.xp.array([
                [1000.0, 1000.0, 1000.0],
                [-1000.0, -1000.0, 0.0]
            ], dtype=self.xp.float32)
            
            output = self.layer.forward(extreme_inputs, training=True)
            
            self.assertFalse(self.xp.isnan(output).any(), "SoftMax suffered an unmitigated NaN explosion!")
            self.assertFalse(self.xp.isinf(output).any(), "SoftMax suffered an overflow infinity leak!")

        def test_forward_does_not_mutate_inputs(self):
            """Ensure forward pass does not modify incoming inputs in-place."""
            inputs = self.xp.array([[1.0, 2.0, 3.0]], dtype=self.xp.float32)
            original_inputs = inputs.copy()

            self.layer.forward(inputs, training=True)

            self.xp.testing.assert_array_equal(inputs, original_inputs)

        def test_backward_does_not_mutate_dvalues(self):
            """Ensure backward pass does not modify incoming upstream gradients in-place."""
            inputs = self.xp.array([[1.0, 2.0, 3.0]], dtype=self.xp.float32)
            self.layer.forward(inputs, training=True)

            dvalues = self.xp.array([[0.1, -0.2, 0.1]], dtype=self.xp.float32)
            original_dvalues = dvalues.copy()

            self.layer.backward(dvalues)

            self.xp.testing.assert_array_equal(dvalues, original_dvalues)
            
        def test_single_sample_batch_size_one(self):
            """Ensure layer handles single-instance mini-batches correctly."""
            inputs = self.xp.array([[0.5, 1.5, -0.5]], dtype=self.xp.float32)
            output = self.layer.forward(inputs, training=False)

            self.assertEqual(output.shape, (1, 3))
            self.assertAlmostEqual(float(self.xp.sum(output)), 1.0, places=5)

        def test_analytical_gradients_limit_definition(self):
            """Validate isolated SoftMax backpropagation against centered finite differences."""

            inputs = self.xp.array([[1.5, 2.5, 0.5], [0.1, -1.2, 3.3]], dtype=self.xp.float32)
            upstream_dvalues = self.xp.array([[0.5, -0.2, 0.1], [1.0, 0.0, -0.5]], dtype=self.xp.float32)
            
            self.layer.forward(inputs, training=True)
            self.layer.backward(upstream_dvalues)
            analytical_dinputs = self.layer.dinputs
            
            epsilon = 1e-4
            numerical_dinputs = self.xp.zeros_like(inputs)
            
            
            def evaluate_loss(x_canvas):
                test_layer = self.layer.__class__()
                if hasattr(test_layer, 'to'):
                    test_layer.to("cupy" if "CUPY" in self.__class__.__name__ else "numpy")
                out_probs = test_layer.forward(x_canvas, training=False)
                return self.xp.sum(out_probs * upstream_dvalues)

            B, C = inputs.shape
            for b in range(B):
                for c in range(C):
                    perturbed_plus = inputs.copy()
                    perturbed_minus = inputs.copy()
                    
                    perturbed_plus[b, c] += epsilon
                    perturbed_minus[b, c] -= epsilon
                    
                    loss_plus = evaluate_loss(perturbed_plus)
                    loss_minus = evaluate_loss(perturbed_minus)
                    
                    numerical_dinputs[b, c] = (loss_plus - loss_minus) / (2.0 * epsilon)
                    

            self.xp.testing.assert_array_almost_equal(
                analytical_dinputs, 
                numerical_dinputs, 
                decimal=3,
                err_msg="Analytical backpropagation pass does not match the limit definition approximation!"
            )

        def test_if_softmax_exempt_from_half_precision_float_flag(self):

            self.layer._precision_exempt = True #set true by default

            raw_inputs = self.xp.array([
                [1.0, 2.0, 3.0, 4.0],
                [-10.0, 0.0, 10.0, 5.0]
            ], dtype=self.xp.float16)

            output_float32 = self.layer.forward(raw_inputs, True)

            self.assertEqual(output_float32.dtype, self.xp.float32)
            
    TestSoftMax.__name__ = class_name
    TestSoftMax.__qualname__ = class_name
    return TestSoftMax

for backend in backends_to_test:
    
    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"

    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)