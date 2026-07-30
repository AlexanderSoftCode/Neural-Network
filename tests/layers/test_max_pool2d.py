import numpy as np
import aether.config as config
from tests.base_case import AetherBaseTestCase

from aether.layers.pooling import MaxPool2d
TARGET_LAYER = MaxPool2d

backends_to_test = ['numpy']
# GPU check & loading loop
try:
    import cupy as cp
    backends_to_test.append('cupy')
except (ImportError, Exception):
    pass  

def make_suite(backend_name, Layer_Class):
    class_name = f"Test_{Layer_Class.__name__}_{backend_name.upper()}"
    class TestMaxPool2d(AetherBaseTestCase):
        FILTER_SIZE = (2, 2)
        STRIDE = (2, 2)
        PADDING = 'valid'
        
        def setUp(self):
            self.backend_name = backend_name
            config.set_backend(backend_name=self.backend_name)
            self.xp = config.xp
            self.as_strided = config.get_stride_utility(self.xp)
            self.test_images = self.xp.random.randn(2, 28, 28, 1)

            self.layer = self.make_layer(
                Layer_Class, filter_size=self.FILTER_SIZE,
                stride=self.STRIDE, padding=self.PADDING,
            )

        # ---- forward pass --------------------------------
        def test_forward_max_valid_shape(self):
            """Verify output dimensions for max pooling with 'valid' padding."""
            output = self.layer.forward(self.test_images, training=True)
            # 28x28 input, 2x2 filter, stride 2, valid padding -> 14x14
            self.assertEqual(output.shape, (2, 14, 14, 1))

        def test_forward_max_same_shape(self):
            """Verify output dimensions for max pooling with 'same' padding
            on an input whose spatial size is not evenly divisible by the
            stride, so the ceil-based 'same' output size actually differs
            from the 'valid' calculation."""
            layer = self.make_layer(Layer_Class, filter_size=self.FILTER_SIZE, stride=self.STRIDE,
                                    padding='same')
            inputs = self.xp.random.randn(2, 7, 7, 3)
            output = layer.forward(inputs, training=True)
            # H_out = ceil(7 / 2) = 4
            self.assertEqual(output.shape, (2, 4, 4, 3))

        def test_forward_max_known_values(self):
            """Verify max pooling selects the correct maximum from each window."""
            layer = self.make_layer(Layer_Class, filter_size=(2, 2), stride=(2, 2),
                                    padding='valid')
            inputs = self.xp.arange(16, dtype=self.xp.float32).reshape(1, 4, 4, 1)
            # [[ 0  1  2  3]
            #  [ 4  5  6  7]
            #  [ 8  9 10 11]
            #  [12 13 14 15]]   
            expected = self.xp.array(
                [[[[5.], [7.]],
                  [[13.], [15.]]]], dtype=self.xp.float32
            )
            output = layer.forward(inputs, training=True)
            self.xp.testing.assert_array_almost_equal(output, expected, decimal=5)

        def test_forward_max_same_padding_preserves_expected_dimensions(self):
            """With stride 1, 'same' padding should preserve spatial dimensions."""
            layer = self.make_layer(Layer_Class, filter_size=(3, 3), stride=(1, 1),
                                    padding='same')
            output = layer.forward(self.test_images, training=True)
            self.assertEqual(output.shape, (2, 28, 28, 1))

        def test_forward_max_valid_padding_reduces_dimensions(self):
            """With stride 1, 'valid' padding should shrink the spatial dimensions."""
            layer = self.make_layer(Layer_Class, filter_size=(3, 3), stride=(1, 1),
                                    padding='valid')
            output = layer.forward(self.test_images, training=True)
            # (28 - 3) / 1 + 1 = 26
            self.assertEqual(output.shape, (2, 26, 26, 1))

        def test_backward_max_non_overlapping_valid(self):
            """Exercise self.strides == self.filter_size branch using 'valid'"""
            output = self.layer.forward(self.test_images, training=True)
            dvalues = self.xp.random.randn(*output.shape)
            dinputs = self.layer.backward(dvalues)
            self.assertEqual(dinputs.shape, self.test_images.shape)

        def test_backward_max_non_overlapping_same(self):
            """Exercise self.strides == self.filter_size branch using 'same'"""
            layer = self.make_layer(Layer_Class, filter_size=self.FILTER_SIZE, stride=self.STRIDE,
                                    padding='same')
            # Use a spatial size that is NOT evenly divisible by the stride so
            # 'same' padding actually inserts real padding, properly
            # exercising the crop-back-to-input-size logic in backward().
            inputs = self.xp.random.randn(2, 7, 7, 1)
            output = layer.forward(inputs, training=True)
            dvalues = self.xp.random.randn(*output.shape)
            dinputs = layer.backward(dvalues)
            self.assertEqual(dinputs.shape, inputs.shape)

        def test_backward_max_routes_gradient_to_maximum(self):
            """ ensure only maximum element in each pooling window recieves the upstream
            graidnet"""
            layer = self.make_layer(Layer_Class, filter_size=self.FILTER_SIZE, stride=self.STRIDE,
                                    padding='valid')
            inputs = self.xp.arange(16, dtype=self.xp.float32).reshape(1, 4, 4, 1)
            layer.forward(inputs, training=True)

            dvalues = self.xp.zeros((1, 2, 2, 1), dtype=self.xp.float32)
            dvalues[0, 0, 0, 0] = 2.0
            dvalues[0, 0, 1, 0] = 3.0
            dvalues[0, 1, 0, 0] = 5.0
            dvalues[0, 1, 1, 0] = 7.0

            dinputs = layer.backward(dvalues)

            expected = self.xp.zeros((1, 4, 4, 1), dtype=self.xp.float32)
            expected[0, 1, 1, 0] = 2.0   # max of top-left window
            expected[0, 1, 3, 0] = 3.0   # max of top-right window
            expected[0, 3, 1, 0] = 5.0   # max of bottom-left window
            expected[0, 3, 3, 0] = 7.0   # max of bottom-right window

            self.xp.testing.assert_array_almost_equal(dinputs, expected, decimal=5)

        # ---- Backward Pass Max Pooling Overlapping Windows ----

        def test_backward_max_overlapping_valid(self):
            layer = self.make_layer(Layer_Class, filter_size=(3, 3), stride=(2, 2),
                                    padding='valid')
            output = layer.forward(self.test_images, training=True)
            dvalues = self.xp.random.randn(*output.shape)
            dinputs = layer.backward(dvalues)
            self.assertEqual(dinputs.shape, self.test_images.shape)

        def test_backward_max_overlapping_same(self):
            layer = self.make_layer(Layer_Class, filter_size=(3, 3), stride=(2, 2),
                                    padding='same')
            output = layer.forward(self.test_images, training=True)
            dvalues = self.xp.random.randn(*output.shape)
            dinputs = layer.backward(dvalues)
            self.assertEqual(dinputs.shape, self.test_images.shape)

        def test_backward_max_overlapping_accumulates_gradients(self):
            layer = self.make_layer(Layer_Class, filter_size=(2, 2), stride=(1, 1),
                                    padding='valid')
            # Center pixel is the maximum of every overlapping window that covers it
            inputs = self.xp.array(
                [[[[1.], [1.], [1.]],
                  [[1.], [9.], [1.]],
                  [[1.], [1.], [1.]]]], dtype=self.xp.float32
            )
            layer.forward(inputs, training=True)

            dvalues = self.xp.ones((1, 2, 2, 1), dtype=self.xp.float32)
            dinputs = layer.backward(dvalues)

            # All four overlapping windows route their gradient to the shared
            # center pixel, so its gradient should be the sum of all four (=4.0)
            expected = self.xp.zeros_like(inputs)
            expected[0, 1, 1, 0] = 4.0
            self.xp.testing.assert_array_almost_equal(dinputs, expected, decimal=5)

        # ---- Validation/Error Handling ----------------------

        def test_forward_invalid_padding_raises_value_error(self):
            layer = self.make_layer(Layer_Class, filter_size=(2, 2), stride=(2, 2),
                                    padding='not_a_real_padding')
            with self.assertRaises(ValueError):
                layer.forward(self.test_images, training=True)

        def test_forward_invalid_input_rank_raises_value_error(self):
            bad_inputs = self.xp.random.randn(28, 28, 1)  # missing batch dim
            with self.assertRaises(ValueError):
                self.layer.forward(bad_inputs, training=True)

        # ---- Numerical Gradient Check ------------------------

        def test_backward_max_numerical_gradient(self):
            """Compare analytical vs numerical using limit def"""
            epsilon = 1e-2
            layer = self.make_layer(Layer_Class, filter_size=(2, 2), stride=(2, 2),
                                    padding='valid')

            fixed_input = self.xp.random.randn(2, 4, 4, 1)
            output = layer.forward(fixed_input, training=True)
            dvalues = self.xp.ones_like(output)  # ones so sum(output) is the scalar loss

            analytical_dinputs = layer.backward(dvalues).copy()

            numerical_dinputs = self.xp.zeros_like(fixed_input)

            S, H, W, C = fixed_input.shape
            for s in range(S):
                for h in range(H):
                    for w in range(W):
                        for c in range(C):
                            fixed_input[s, h, w, c] += epsilon
                            loss_plus = self.xp.sum(layer.forward(fixed_input, training=False))

                            fixed_input[s, h, w, c] -= 2 * epsilon
                            loss_minus = self.xp.sum(layer.forward(fixed_input, training=False))

                            fixed_input[s, h, w, c] += epsilon

                            numerical_dinputs[s, h, w, c] = (loss_plus - loss_minus) / (2 * epsilon)

            self.xp.testing.assert_array_almost_equal(
                analytical_dinputs, numerical_dinputs, decimal=3
            )

    TestMaxPool2d.__name__ = class_name
    TestMaxPool2d.__qualname__ = class_name

    return TestMaxPool2d

for backend in backends_to_test:
    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"
    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)