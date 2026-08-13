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
            
            self.layer = self.make_built_layer(
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
            layer = self.make_built_layer(
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
            layer = self.make_built_layer(
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
            
            self.xp.testing.assert_array_almost_equal(output, expected_output)

        def test_conv_stride_length(self):

            layer = self.make_built_layer(
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
            layer = self.make_built_layer(
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
            layer = self.make_built_layer(
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
        
                            # f(x + h) -- mutate the buffer in place, then invalidate,
                            # exactly like a fused optimizer kernel writing into
                            # layer.weights and calling invalidate_shadow_caches().
                            layer.filter_weights[fh, fw, c, f] += epsilon
                            layer.invalidate_shadow_caches()
                            layer.forward(fixed_input, training=False)
                            loss_plus = self.xp.sum(layer.output)
        
                            # f(x - h)
                            layer.filter_weights[fh, fw, c, f] -= 2 * epsilon
                            layer.invalidate_shadow_caches()
                            layer.forward(fixed_input, training=False)
                            loss_minus = self.xp.sum(layer.output)
        
                            # Restore -- also needs invalidating, so the layer is
                            # left in a correct state for the next element's pass
                            # (and for anything run after this loop).
                            layer.filter_weights[fh, fw, c, f] += epsilon
                            layer.invalidate_shadow_caches()
        
                            numerical_dweights[fh, fw, c, f] = (loss_plus - loss_minus) / (2 * epsilon)
        
            self.xp.testing.assert_array_almost_equal(
                analytical_dweights, numerical_dweights, decimal=2
            )

        def test_conv_multichannel_input(self):
            """Verify forward pass with multi-channel input (e.g., RGB with C_in=3)"""
            layer = self.make_built_layer(
                Layer_Class,
                in_channels=3,
                out_channels=8,
                filter_size=self.FILTER_SIZE,
                stride=self.STRIDE,
                padding=self.PADDING
            )
            rgb_images = self.xp.random.randn(2, 28, 28, 3)
            output = layer.forward(rgb_images, training=False)
            self.assertEqual(output.shape, (2, 28, 28, 8))

        def test_conv_variable_batch_sizes(self):
            """Verify shape cache handles dynamic batch sizes without re-compilation or shape errors"""
            inputs_b4 = self.xp.random.randn(4, 16, 16, self.IN_CHANNELS)
            out_b4 = self.layer.forward(inputs_b4, training=False)
            self.assertEqual(out_b4.shape, (4, 16, 16, self.OUT_CHANNELS))

            inputs_b1 = self.xp.random.randn(1, 16, 16, self.IN_CHANNELS)
            out_b1 = self.layer.forward(inputs_b1, training=False)
            self.assertEqual(out_b1.shape, (1, 16, 16, self.OUT_CHANNELS))

        def test_conv_non_square_filter(self):
            """Verify rectangular filter geometries such as (5, 3)"""
            layer = self.make_built_layer(
                Layer_Class,
                in_channels=self.IN_CHANNELS,
                out_channels=self.OUT_CHANNELS,
                filter_size=(5, 3),
                stride=(1, 1),
                padding="same"
            )
            output = layer.forward(self.test_images, training=False)
            self.assertEqual(output.shape, (2, 28, 28, self.OUT_CHANNELS))

        def test_fp16_cache_invalidation_on_weight_update(self):
            """Ensure updating layer weights invalidates the FP16 shadow cache"""
            if self.backend_name == 'cupy':
                self.layer._compile_for_device('cupy')
                _ = self.layer.forward(self.test_images, training=False)
                
                # If GPU matrix-core path was taken, cache should now be valid
                if self.layer._fp16_weight_cache is not None:
                    self.assertTrue(self.layer._fp16_weight_valid)

                    # Update weights via property setter
                    new_weights = self.xp.random.randn(*self.layer.filter_weights.shape).astype(self.xp.float32)
                    self.layer.weights = new_weights
                    self.assertFalse(self.layer._fp16_weight_valid)

        def test_forward_gpu_fallback_numeric_parity(self):
            """Verify numerical equivalence between _forward_gpu and _forward_fallback (CuPy backend only)"""
            if self.backend_name == 'cupy':
                from aether.custom_kernels import conv_kernel
                if conv_kernel.get_is_conv_gpu_available():
                    fixed_input = self.xp.random.randn(2, 14, 14, self.IN_CHANNELS).astype(self.xp.float32)

                    # Execute GPU matrix-core path
                    gpu_out = self.layer._forward_gpu(fixed_input, training=False)

                    # Execute CPU/CuPy fallback path
                    fallback_out = self.layer._forward_fallback(fixed_input, training=False)

                    # FP16 Tensor Cores introduce minor precision differences vs FP32 einsum
                    self.xp.testing.assert_allclose(gpu_out, fallback_out, rtol=1e-2, atol=1e-2)

    TestConvLayer.__name__ = class_name
    TestConvLayer.__qualname__ = class_name
    return TestConvLayer

for backend in backends_to_test:
    
    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"
    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)