import aether.config as config
from tests.base_case import AetherBaseTestCase

from aether.layers.conv import Conv
TARGET_LAYER = Conv

backends_to_test = ['numpy']
try:
    import cupy as cp
    backends_to_test.append('cupy')

except (ImportError, ModuleNotFoundError):
    pass  


def make_suite(backend_name, Layer_Class):

    class_name = f"Test_{TARGET_LAYER.__name__}_{backend_name.upper()}"
    class TestConvLayer(AetherBaseTestCase):
        IN_CHANNELS=1
        OUT_CHANNELS=4
        FILTER_SIZE=(3,3)
        STRIDE=(1,1)
        PADDING="same"
        def setUp(self):

            self.backend_name = backend_name
            config.set_backend(backend_name=self.backend_name)
            self.xp = config.xp
            self.as_strided = config.get_stride_utility(self.xp)
            
            self.layer = self.make_layer(
                Layer_Class, 
                in_channels=self.IN_CHANNELS, 
                out_channels=self.OUT_CHANNELS, 
                filter_size=self.FILTER_SIZE,
                stride=self.STRIDE,
                padding=self.PADDING
            )
            #Dummy image, batch of 2, 28x28 pixels, 1 channel
            self.test_images = self.xp.random.randn(2, 28, 28, 1)

        def test_conv_forward_shape(self):
            """Verify output dimensions based on padding and stride"""
            output = self.layer.forward(self.test_images, training=True)
            # Given padding = 1, kernel = 3, 28x28 remainds 28x28 (same)
            self.assertEqual(output.shape, (2, 28, 28, 4))
        
        def test_parameters_initialization(self):
            """Compare filter weights and filter bias shapes"""
            self.assertEqual(self.layer.filter_weights.shape, (3, 3, 1, 4))
            self.assertEqual(self.layer.biases.shape, (4,))

        def test_backward_gradient_shapes(self):
            """Verify backprop gradients match weight shapes"""
            output = self.layer.forward(self.test_images, training=True)
            dvalues = self.xp.random.randn(*output.shape) #mock gradient
            self.layer.backward(dvalues)

            self.assertEqual(self.layer.dweights.shape, self.layer.filter_weights.shape)
            self.assertEqual(self.layer.dbiases.shape, self.layer.biases.shape)

        def test_conv_same_padding_stride(self):
            # 28x28 input, 3x3 filter, stride=(2,2), padding='same'
            layer = self.make_layer(
                Layer_Class, 
                in_channels=self.IN_CHANNELS, 
                out_channels=self.OUT_CHANNELS, 
                filter_size=self.FILTER_SIZE,
                stride=(2,2),
                padding=self.PADDING
            )
            output = layer.forward(self.test_images, training=True)
            
            # Calculation: P=1 -> int((28 + 2 - 3)/2 + 1) = 14
            self.assertEqual(output.shape, (2, 14, 14, 4))
            
        def test_conv_valid_padding_stride(self):
            # 28x28 input, 3x3 filter, stride = (2,2), valid padding
            layer = self.make_layer(
                Layer_Class, 
                in_channels=self.IN_CHANNELS, 
                out_channels=self.OUT_CHANNELS, 
                filter_size=self.FILTER_SIZE,
                stride=(2,2),
                padding="valid"
            )
            output = layer.forward(self.test_images, training=True)
            # (28 - 3) / 2 + 1 = 13.5 -> floor to 13
            self.assertEqual(output.shape, (2, 13, 13, 4))

        def test_conv_zero_input(self):
            "Given an input of all zeros, show an output of all zeros (plus biases)"

            zero_input = self.xp.zeros((1, 28, 28, 1))
            
            # Expected: Every pixel in the output should equal the bias for that filter
            # We reshape biases to (1, 1, 1, num_filters) to match output shape
            
            output = self.layer.forward(zero_input, training = False)
            expected_output = self.xp.broadcast_to(self.layer.biases, output.shape)
            
            #compare the values in the numpy arrays
            self.xp.testing.assert_array_almost_equal(output, expected_output)

        def test_conv_stride_length(self):

            layer = self.make_layer(
                Layer_Class,
                in_channels=self.IN_CHANNELS,
                out_channels=self.OUT_CHANNELS,
                filter_size=self.FILTER_SIZE,
                stride=(2,1),
                padding=self.PADDING
            )
            output = layer.forward(self.test_images, training = False)
            
            self.assertEqual(output.shape, (2, 14, 28, 4))

        def test_conv_stride_width(self):
            layer = self.make_layer(
                Layer_Class,
                in_channels=self.IN_CHANNELS,
                out_channels=self.OUT_CHANNELS,
                filter_size=self.FILTER_SIZE,
                stride=(1,2),
                padding=self.PADDING
            )
            output = layer.forward(self.test_images, training=False)
            self.assertEqual(output.shape, (2, 28, 14, 4))

        def test_conv_numerical_gradient_check(self):
            """
            Finite difference check: verify analytical dweights matches numerical approximation
            for the Conv. Uses a small input/filter to keep the double loop feasible.
            f'(x) ≈ (f(x + h) - f(x - h)) / 2h
            """
            epsilon = 1e-2
            layer = self.make_layer(
                Layer_Class,
                in_channels=self.IN_CHANNELS,
                out_channels=2,
                filter_size=self.FILTER_SIZE,
                stride=self.STRIDE,
                padding=self.PADDING
            )

            fixed_input = self.xp.random.randn(2, 8, 8, 1)  # batch=2, 8x8, 1 channel
            dvalues = self.xp.ones((2, 8, 8, 2))             # ones so sum(output) is the scalar loss

            # Analytical gradient
            layer.forward(fixed_input, training=False)
            layer.backward(dvalues)
            analytical_dweights = layer.dweights.copy()

            # Numerical gradient — iterate over every element of filter_weights (3x3x1x2 = 18)
            numerical_dweights = self.xp.zeros_like(layer.filter_weights)

            for fh in range(layer.filter_weights.shape[0]):
                for fw in range(layer.filter_weights.shape[1]):
                    for c in range(layer.filter_weights.shape[2]):
                        for f in range(layer.filter_weights.shape[3]):

                            # f(x + h)
                            layer.filter_weights[fh, fw, c, f] += epsilon
                            layer.forward(fixed_input, training=False)
                            loss_plus = self.xp.sum(layer.output)

                            # f(x - h)
                            layer.filter_weights[fh, fw, c, f] -= 2 * epsilon
                            layer.forward(fixed_input, training=False)
                            loss_minus = self.xp.sum(layer.output)

                            # Restore
                            layer.filter_weights[fh, fw, c, f] += epsilon

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