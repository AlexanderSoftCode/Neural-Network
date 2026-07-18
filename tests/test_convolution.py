import unittest
import numpy as np 

# Core Framework Configuration Imports
from aether.config import set_backend, get_stride_utility
import aether.config as config
from tests.base_case import AetherBaseTestCase

from aether.blocks.conv import Conv
TARGET_LAYER = Conv

backends_to_test = ['numpy']
try:
    import cupy as cp
    backends_to_test.append('cupy')

except (ImportError, Exception):
    pass  


def make_suite(backend_name, Layer_Class):

    class_name = f"Test_{TARGET_LAYER.__name__}_{backend_name.upper()}"
    class TestConvLayer(AetherBaseTestCase):

        def setUp(self):

            set_backend(backend_name=backend_name)
            self.xp = config.xp
            self.as_strided = get_stride_utility(self.xp)
            
            self.conv = Layer_Class(input_shape=(1, 1, 1), num_filters=4,
                                filter_size = (3, 3), strides=(1,1), padding='same')
            #Dummy image, batch of 2, 28x28 pixels, 1 channel
            self.test_images = self.xp.random.randn(2, 28, 28, 1)

        def test_conv_forward_shape(self):
            """Verify output dimensions based on padding and stride"""
            output = self.conv.forward(self.test_images, training=True)
            # Given padding = 1, kernel = 3, 28x28 remainds 28x28 (same)
            self.assertEqual(output.shape, (2, 28, 28, 4))
        
        def test_parameters_initialization(self):
            """Compare filter weights and filter bias shapes"""
            self.assertEqual(self.conv.filter_weights.shape, (3, 3, 1, 4))
            self.assertEqual(self.conv.biases.shape, (4,))

        def test_backward_gradient_shapes(self):
            """Verify backprop gradients match weight shapes"""
            output = self.conv.forward(self.test_images, training=True)
            dvalues = self.xp.random.randn(*output.shape) #mock gradient
            self.conv.backward(dvalues)

            self.assertEqual(self.conv.dweights.shape, self.conv.filter_weights.shape)
            self.assertEqual(self.conv.dbiases.shape, self.conv.biases.shape)

        def test_conv_same_padding_stride(self):
            # 28x28 input, 3x3 filter, stride=(2,2), padding='same'
            conv = Conv(input_shape=(28, 28, 1), num_filters=4, 
                            filter_size=(3, 3), strides=(2, 2), padding='same')
            output = conv.forward(self.test_images, training=True)
            
            # Calculation: P=1 -> int((28 + 2 - 3)/2 + 1) = 14
            self.assertEqual(output.shape, (2, 14, 14, 4))
            
        def test_conv_valid_padding_stride(self):
            # 28x28 input, 3x3 filter, stride = (2,2), valid padding

            conv = Conv(input_shape=(28,28,1), num_filters=4, filter_size=(3,3),
                            strides=(2,2), padding='valid')
            output = conv.forward(self.test_images, training=True)
            # (28 - 3) / 2 + 1 = 13.5 -> floor to 13
            self.assertEqual(output.shape, (2, 13, 13, 4))

        def test_conv_zero_input(self):
            "Given an input of all zeros, show an output of all zeros (plus biases)"

            conv = Conv(input_shape=(28,28,1), num_filters=4, filter_size=(3,3),
                            strides=(2,2), padding='same')
            zero_input = self.xp.zeros((1, 28, 28, 1))
            
            # Expected: Every pixel in the output should equal the bias for that filter
            # We reshape biases to (1, 1, 1, num_filters) to match output shape
            
            output = conv.forward(zero_input, training = False)
            expected_output = self.xp.broadcast_to(conv.biases, output.shape)
            
            #compare the values in the numpy arrays
            self.xp.testing.assert_array_almost_equal(output, expected_output)

        def test_conv_stride_length(self):
            
            conv = Conv(input_shape=(28,28,1), num_filters=4, filter_size=(3,3),
                            strides=(2,1), padding='same')
            output = conv.forward(self.test_images, training = False)
            
            self.assertEqual(output.shape, (2, 14, 28, 4))
        def test_conv_numerical_gradient_check(self):
            """
            Finite difference check: verify analytical dweights matches numerical approximation
            for the Conv. Uses a small input/filter to keep the double loop feasible.
            f'(x) ≈ (f(x + h) - f(x - h)) / 2h
            """
            epsilon = 1e-2

            conv = Conv(
                input_shape=(8, 8, 1),
                num_filters=2,
                filter_size=(3, 3),
                strides=(1, 1),
                padding='same'
            )

            fixed_input = self.xp.random.randn(2, 8, 8, 1)  # batch=2, 8x8, 1 channel
            dvalues = self.xp.ones((2, 8, 8, 2))             # ones so sum(output) is the scalar loss

            # Analytical gradient
            conv.forward(fixed_input, training=False)
            conv.backward(dvalues)
            analytical_dweights = conv.dweights.copy()

            # Numerical gradient — iterate over every element of filter_weights (3x3x1x2 = 18)
            numerical_dweights = self.xp.zeros_like(conv.filter_weights)

            for fh in range(conv.filter_weights.shape[0]):
                for fw in range(conv.filter_weights.shape[1]):
                    for c in range(conv.filter_weights.shape[2]):
                        for f in range(conv.filter_weights.shape[3]):

                            # f(x + h)
                            conv.filter_weights[fh, fw, c, f] += epsilon
                            conv.forward(fixed_input, training=False)
                            loss_plus = self.xp.sum(conv.output)

                            # f(x - h)
                            conv.filter_weights[fh, fw, c, f] -= 2 * epsilon
                            conv.forward(fixed_input, training=False)
                            loss_minus = self.xp.sum(conv.output)

                            # Restore
                            conv.filter_weights[fh, fw, c, f] += epsilon

                            numerical_dweights[fh, fw, c, f] = (loss_plus - loss_minus) / (2 * epsilon)

            self.xp.testing.assert_array_almost_equal(
                analytical_dweights, numerical_dweights, decimal=3
            )

    TestConvLayer.__name__ = class_name
    TestConvLayer.__qualname__ = class_name
    return TestConvLayer

# Global Unpacking Loop
for backend in backends_to_test:
    
    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"
    # Bind generated class to the global namespace for unittest discovery
    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)