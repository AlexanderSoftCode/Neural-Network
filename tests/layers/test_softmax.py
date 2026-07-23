import aether.config as config
from tests.base_case import AetherBaseTestCase

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

    class TestSoftMax(AetherBaseTestCase):
        def setUp(self):

            config.set_backend(backend_name=backend_name)
            self.xp = config.xp

            self.layer = Layer_Class()

            if hasattr(self.layer, 'to'):
                self.layer.to(device=backend_name)
            elif hasattr(self.layer, '_compile_for_device'):
                self.layer._compile_for_device(backend_name) 
    
        def test_backend_pointer_swap(self):
            """Verify the layer successfully routes to the correct backend execution hooks."""
            # Ensure global config module matches the expected test string context
            if "CUPY" in self.__class__.__name__:
                self.assertEqual(self.xp.__name__, "cupy")
            else:
                self.assertEqual(self.xp.__name__, "numpy")

        def test_forward_probabilities(self):
            """Verify SoftMax bounds values between 0 and 1, and rows sum to 1.0."""
            # Generate arbitrary unnormalized logits
            raw_inputs = self.xp.array([
                [1.0, 2.0, 3.0, 4.0],
                [-10.0, 0.0, 10.0, 5.0]
            ], dtype=self.xp.float32)
            
            # Execute forward pass
            output = self.layer.forward(raw_inputs, training=True)
            
            # Assert row summation rules polymorphically via active namespace
            row_sums = self.xp.sum(output, axis=1)
            expected_sums = self.xp.ones(row_sums.shape, dtype=self.xp.float32)
            
            self.xp.testing.assert_array_almost_equal(row_sums, expected_sums, decimal=5)
            self.assertTrue(self.xp.all(output >= 0.0) and self.xp.all(output <= 1.0))

        def test_numerical_stability(self):
            """Verify that extreme logit input scales do not explode or result in NaNs."""
            # Mix huge values with tiny ones (triggers overflow if not stabilized)
            extreme_inputs = self.xp.array([
                [1000.0, 1000.0, 1000.0],
                [-1000.0, -1000.0, 0.0]
            ], dtype=self.xp.float32)
            
            output = self.layer.forward(extreme_inputs, training=True)
            
            self.assertFalse(self.xp.isnan(output).any(), "SoftMax suffered an unmitigated NaN explosion!")
            self.assertFalse(self.xp.isinf(output).any(), "SoftMax suffered an overflow infinity leak!")

        def test_analytical_gradients_limit_definition(self):
            """Validate isolated SoftMax backpropagation against centered finite differences."""
            # Setup baseline data layout
            inputs = self.xp.array([[1.5, 2.5, 0.5], [0.1, -1.2, 3.3]], dtype=self.xp.float32)
            upstream_dvalues = self.xp.array([[0.5, -0.2, 0.1], [1.0, 0.0, -0.5]], dtype=self.xp.float32)
            
            # 1. Run Analytical Pass (Your framework's high-speed vectorized code)
            self.layer.forward(inputs, training=True)
            self.layer.backward(upstream_dvalues)
            analytical_dinputs = self.layer.dinputs
            
            # 2. Compute Numerical Approximation via Center Difference Limit Method
            epsilon = 1e-4
            numerical_dinputs = self.xp.zeros_like(inputs)
            
            # Define a helper to evaluate the "Loss Function" value: Upstream Gradient Dot Output
            def evaluate_loss(x_canvas):
                # We instantiate a clean evaluation pass to isolate internal cache states
                test_layer = self.layer.__class__()
                if hasattr(test_layer, 'to'):
                    test_layer.to("cupy" if "CUPY" in self.__class__.__name__ else "numpy")
                out_probs = test_layer.forward(x_canvas, training=False)
                return self.xp.sum(out_probs * upstream_dvalues)

            # Sequentially perturb each coordinate element to simulate the limit approach
            B, C = inputs.shape
            for b in range(B):
                for c in range(C):
                    # Create isolated canvases for positive and negative epsilon shifts
                    perturbed_plus = inputs.copy()
                    perturbed_minus = inputs.copy()
                    
                    perturbed_plus[b, c] += epsilon
                    perturbed_minus[b, c] -= epsilon
                    
                    # Evaluate scalar loss differences
                    loss_plus = evaluate_loss(perturbed_plus)
                    loss_minus = evaluate_loss(perturbed_minus)
                    
                    # Center difference approximation rule
                    numerical_dinputs[b, c] = (loss_plus - loss_minus) / (2.0 * epsilon)
                    

            self.xp.testing.assert_array_almost_equal(
                analytical_dinputs, 
                numerical_dinputs, 
                decimal=3,
                err_msg="Analytical backpropagation pass does not match the limit definition approximation!"
            )

    TestSoftMax.__name__ = class_name
    TestSoftMax.__qualname__ = class_name
    return TestSoftMax

# Global Unpacking Loop
for backend in backends_to_test:
    
    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"

    # Bind generated class to the global namespace for unittest discovery
    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)