import unittest
import numpy as np

from aether.config import set_backend
import aether.config as config
from tests.base_case import AetherBaseTestCase

from aether.blocks.linear import Dense
TARGET_LAYER = Dense

backends_to_test = ['numpy']

try:
    import cupy as cp
    backends_to_test.append('cupy')
except:
    pass 

def make_suite(backend_name, Layer_Class):    

    class_name = f"Test_{Layer_Class.__name__}_{backend_name.upper()}"
    class TestDenseLayer(unittest.TestCase): 
        def setUp(self):
            set_backend(backend_name=backend_name)
            self.xp = config.xp

            self.xp.random.seed(42)
            self.layer = Layer_Class(n_inputs=5, n_neurons=3)
            self.test_input = self.xp.random.randn(10, 5) 

        def test_forward_shape(self):
            "Verify output shape is 10, 3"
            output = self.layer.forward(inputs=self.test_input, training=True)
            self.assertEqual(output.shape, (10, 3))
        
        def test_parameters_initialization(self): 
            """Now check if weights and biases are initialized correctly"""

            self.assertEqual(self.layer.weights.shape, (5, 3))
            self.assertEqual(self.layer.biases.shape, (1, 3))
        
            """Verify He initialization scale approximately matches sqrt(2/n) """
            expected_std = self.xp.sqrt(2 / 5) * 0.01
            actual_std = self.xp.std(self.layer.weights)

            self.assertAlmostEqual(actual_std, expected_std, delta = 0.1) 

        def test_backward_gradient_shapes(self):
            """Ensure backprop produces gradients matching parameter shapes"""

            output = self.layer.forward(self.test_input, training=True)
            # Mock dvalues
            dvalues = self.xp.random.randn(*output.shape)
            self.layer.backward(dvalues)

            self.assertEqual(self.layer.dweights.shape, self.layer.weights.shape)
            self.assertEqual(self.layer.dbiases.shape, self.layer.biases.shape)

        def test_dense_regularization(self):

            layer_reg = Dense(5, 3, weight_regularizer_l2=0.1)
            layer_reg.forward(self.test_input, training=True)
            layer_reg.backward(self.xp.ones((10, 3)))

            layer_no_reg = Dense(5, 3, weight_regularizer_l2=0)

            layer_no_reg.weights = layer_reg.weights.copy()
            layer_no_reg.biases = layer_reg.biases.copy()

            layer_no_reg.forward(self.test_input, training=True)
            layer_no_reg.backward(self.xp.ones((10, 3)))

            self.assertFalse(self.xp.array_equal(layer_reg.dweights, layer_no_reg.dweights))

            expected_penalty = 2 * 0.1 * layer_reg.weights
            actual_diff = layer_reg.dweights - layer_no_reg.dweights
            self.xp.testing.assert_array_almost_equal(actual_diff, expected_penalty)

        def test_dense_zero_input(self):
            "Given an input of all zeros, show an output of all zeros (plus biases)"

            layer = Dense(n_inputs = 728, n_neurons = 20)
            
            zero_input = self.xp.zeros((1, 728))

            # Every pixel in the output should equal the bias for that layer
            # We reshape the biases to (1, biases) to match output shape

            output = layer.forward(zero_input, training = False)
            expected_output = self.xp.broadcast_to(layer.biases, output.shape)

            # compare the values
            self.xp.testing.assert_array_almost_equal(output, expected_output)
        def test_numerical_gradient_check(self):
            """
            Finite difference check: verify analytical dweights matches numerical approximation.
            f'(x) ≈ (f(x + h) - f(x - h)) / 2h
            """
            epsilon = 1e-2
            layer = Dense(n_inputs=5, n_neurons=3)

            fixed_input = self.xp.random.randn(4, 5)
            dvalues = self.xp.ones((4, 3))

            # Analytical gradient
            layer.forward(fixed_input, training=False)
            layer.backward(dvalues)
            analytical_dweights = layer.dweights.copy()

            # Numerical gradient
            numerical_dweights = self.xp.zeros_like(layer.weights)

            for i in range(layer.weights.shape[0]):
                for j in range(layer.weights.shape[1]):

                    # f(x + h)
                    layer.weights[i, j] += epsilon
                    layer.forward(fixed_input, training=False)
                    loss_plus = self.xp.sum(layer.output)

                    # f(x - h)
                    layer.weights[i, j] -= 2 * epsilon
                    layer.forward(fixed_input, training=False)
                    loss_minus = self.xp.sum(layer.output)

                    # Restore weight
                    layer.weights[i, j] += epsilon

                    numerical_dweights[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

            self.xp.testing.assert_array_almost_equal(
                analytical_dweights, numerical_dweights, decimal=4
            )
    TestDenseLayer.__name__ = class_name
    TestDenseLayer.__qualname__ = class_name
    return TestDenseLayer 

for backend in backends_to_test:

    class_name = f"Test_{TARGET_LAYER}.__name__)_{backend.upper()}"

    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)