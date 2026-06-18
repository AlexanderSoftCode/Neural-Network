import unittest
import numpy as np

from CNN.models.CNN_classes import Conv_Layer as Conv_Layer_CPU
from CNN.models.CNN_classes import Layer_Dense as Layer_Dense_CPU

test_profiles = [
    (np, Conv_Layer_CPU, Layer_Dense_CPU)
]

try:
    import cupy as cp
    # By placing this import here, any driver initialization errors or compilation 
    # errors caused by the global ElementwiseKernel will be caught safely.
    from CNN.models.CNN_classes_cupy import Conv_Layer as Conv_Layer_GPU
    from CNN.models.CNN_classes_cupy import Layer_Dense as Layer_Dense_GPU
    
    test_profiles.append((cp, Conv_Layer_GPU, Layer_Dense_GPU))
except Exception:
    # Safely fall back if CuPy or a valid GPU execution context is missing
    pass

def make_suite(xp, Layer_Dense, Conv_Layer):    
    class TestDenseLayer(unittest.TestCase): 
        def setUp(self):
            xp.random.seed(42)
            self.layer = Layer_Dense(n_inputs=5, n_neurons=3)
            self.test_input = xp.random.randn(10, 5) 

        def test_forward_shape(self):
            "Verify output shape is 10, 3"
            output = self.layer.forward(inputs=self.test_input, training=True)
            self.assertEqual(output.shape, (10, 3))
        
        def test_parameters_initialization(self): 
            """Now check if weights and biases are initialized correctly"""

            self.assertEqual(self.layer.weights.shape, (5, 3))
            self.assertEqual(self.layer.biases.shape, (1, 3))
        
            """Verify He initialization scale approximately matches sqrt(2/n) [cite: 52, 53]"""
            expected_std = xp.sqrt(2 / 5) * 0.01
            actual_std = xp.std(self.layer.weights)

            self.assertAlmostEqual(actual_std, expected_std, delta = 0.1) 

        def test_backward_gradient_shapes(self):
            """Ensure backprop produces gradients matching parameter shapes"""

            output = self.layer.forward(self.test_input, training=True)
            # Mock dvalues
            dvalues = xp.random.randn(*output.shape)
            self.layer.backward(dvalues)

            self.assertEqual(self.layer.dweights.shape, self.layer.weights.shape)
            self.assertEqual(self.layer.dbiases.shape, self.layer.biases.shape)

        def test_dense_regularization(self):

            layer_reg = Layer_Dense(5, 3, weight_regularizer_l2=0.1)
            layer_reg.forward(self.test_input, training=True)
            layer_reg.backward(xp.ones((10, 3)))

            layer_no_reg = Layer_Dense(5, 3, weight_regularizer_l2=0)

            layer_no_reg.weights = layer_reg.weights.copy()
            layer_no_reg.biases = layer_reg.biases.copy()

            layer_no_reg.forward(self.test_input, training=True)
            layer_no_reg.backward(xp.ones((10, 3)))

            self.assertFalse(xp.array_equal(layer_reg.dweights, layer_no_reg.dweights))

            expected_penalty = 2 * 0.1 * layer_reg.weights
            actual_diff = layer_reg.dweights - layer_no_reg.dweights
            xp.testing.assert_array_almost_equal(actual_diff, expected_penalty)

        def test_dense_zero_input(self):
            "Given an input of all zeros, show an output of all zeros (plus biases)"

            layer = Layer_Dense(n_inputs = 728, n_neurons = 20)
            
            zero_input = xp.zeros((1, 728))

            # Every pixel in the output should equal the bias for that layer
            # We reshape the biases to (1, biases) to match output shape

            output = layer.forward(zero_input, training = False)
            expected_output = xp.broadcast_to(layer.biases, output.shape)

            # compare the values
            xp.testing.assert_array_almost_equal(output, expected_output)
        def test_numerical_gradient_check(self):
            """
            Finite difference check: verify analytical dweights matches numerical approximation.
            f'(x) ≈ (f(x + h) - f(x - h)) / 2h
            """
            epsilon = 1e-2
            layer = Layer_Dense(n_inputs=5, n_neurons=3)

            fixed_input = xp.random.randn(4, 5)
            dvalues = xp.ones((4, 3))

            # Analytical gradient
            layer.forward(fixed_input, training=False)
            layer.backward(dvalues)
            analytical_dweights = layer.dweights.copy()

            # Numerical gradient
            numerical_dweights = xp.zeros_like(layer.weights)

            for i in range(layer.weights.shape[0]):
                for j in range(layer.weights.shape[1]):

                    # f(x + h)
                    layer.weights[i, j] += epsilon
                    layer.forward(fixed_input, training=False)
                    loss_plus = xp.sum(layer.output)

                    # f(x - h)
                    layer.weights[i, j] -= 2 * epsilon
                    layer.forward(fixed_input, training=False)
                    loss_minus = xp.sum(layer.output)

                    # Restore weight
                    layer.weights[i, j] += epsilon

                    numerical_dweights[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

            xp.testing.assert_array_almost_equal(
                analytical_dweights, numerical_dweights, decimal=4
            )
    class TestConvLayer(unittest.TestCase):

        def setUp(self):
            self.conv = Conv_Layer(input_shape=(1, 1, 1), num_filters=4,
                                filter_size = (3, 3), strides=(1,1), padding='same')
            #Dummy image, batch of 2, 28x28 pixels, 1 channel
            self.test_images = xp.random.randn(2, 28, 28, 1)

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
            dvalues = xp.random.randn(*output.shape) #mock gradient
            self.conv.backward(dvalues)

            self.assertEqual(self.conv.dweights.shape, self.conv.filter_weights.shape)
            self.assertEqual(self.conv.dbiases.shape, self.conv.biases.shape)

        def test_conv_same_padding_stride(self):
            # 28x28 input, 3x3 filter, stride=(2,2), padding='same'
            conv = Conv_Layer(input_shape=(28, 28, 1), num_filters=4, 
                            filter_size=(3, 3), strides=(2, 2), padding='same')
            output = conv.forward(self.test_images, training=True)
            
            # Calculation: P=1 -> int((28 + 2 - 3)/2 + 1) = 14
            self.assertEqual(output.shape, (2, 14, 14, 4))
            
        def test_conv_valid_padding_stride(self):
            # 28x28 input, 3x3 filter, stride = (2,2), valid padding

            conv = Conv_Layer(input_shape=(28,28,1), num_filters=4, filter_size=(3,3),
                            strides=(2,2), padding='valid')
            output = conv.forward(self.test_images, training=True)
            # (28 - 3) / 2 + 1 = 13.5 -> floor to 13
            self.assertEqual(output.shape, (2, 13, 13, 4))

        def test_conv_zero_input(self):
            "Given an input of all zeros, show an output of all zeros (plus biases)"

            conv = Conv_Layer(input_shape=(28,28,1), num_filters=4, filter_size=(3,3),
                            strides=(2,2), padding='same')
            zero_input = xp.zeros((1, 28, 28, 1))
            
            # Expected: Every pixel in the output should equal the bias for that filter
            # We reshape biases to (1, 1, 1, num_filters) to match output shape
            
            output = conv.forward(zero_input, training = False)
            expected_output = xp.broadcast_to(conv.biases, output.shape)
            
            #compare the values in the numpy arrays
            xp.testing.assert_array_almost_equal(output, expected_output)

        def test_conv_stride_length(self):
            
            conv = Conv_Layer(input_shape=(28,28,1), num_filters=4, filter_size=(3,3),
                            strides=(2,1), padding='same')
            output = conv.forward(self.test_images, training = False)
            
            self.assertEqual(output.shape, (2, 14, 28, 4))
        def test_conv_numerical_gradient_check(self):
            """
            Finite difference check: verify analytical dweights matches numerical approximation
            for the Conv_Layer. Uses a small input/filter to keep the double loop feasible.
            f'(x) ≈ (f(x + h) - f(x - h)) / 2h
            """
            epsilon = 1e-2

            conv = Conv_Layer(
                input_shape=(8, 8, 1),
                num_filters=2,
                filter_size=(3, 3),
                strides=(1, 1),
                padding='same'
            )

            fixed_input = xp.random.randn(2, 8, 8, 1)  # batch=2, 8x8, 1 channel
            dvalues = xp.ones((2, 8, 8, 2))             # ones so sum(output) is the scalar loss

            # Analytical gradient
            conv.forward(fixed_input, training=False)
            conv.backward(dvalues)
            analytical_dweights = conv.dweights.copy()

            # Numerical gradient — iterate over every element of filter_weights (3x3x1x2 = 18)
            numerical_dweights = xp.zeros_like(conv.filter_weights)

            for fh in range(conv.filter_weights.shape[0]):
                for fw in range(conv.filter_weights.shape[1]):
                    for c in range(conv.filter_weights.shape[2]):
                        for f in range(conv.filter_weights.shape[3]):

                            # f(x + h)
                            conv.filter_weights[fh, fw, c, f] += epsilon
                            conv.forward(fixed_input, training=False)
                            loss_plus = xp.sum(conv.output)

                            # f(x - h)
                            conv.filter_weights[fh, fw, c, f] -= 2 * epsilon
                            conv.forward(fixed_input, training=False)
                            loss_minus = xp.sum(conv.output)

                            # Restore
                            conv.filter_weights[fh, fw, c, f] += epsilon

                            numerical_dweights[fh, fw, c, f] = (loss_plus - loss_minus) / (2 * epsilon)

            xp.testing.assert_array_almost_equal(
                analytical_dweights, numerical_dweights, decimal=4
            )
    return TestDenseLayer, TestConvLayer

for _xp, _conv, _dense in test_profiles:
    dense_layer_suite, conv_layer_suite = make_suite(
        xp=_xp, 
        Layer_Dense=_dense, 
        Conv_Layer=_conv
    )

    backend_name = _xp.__name__

    globals()[f"TestDenseLayer_{backend_name}"] = dense_layer_suite
    globals()[f"TestConvLayer_{backend_name}"] = conv_layer_suite