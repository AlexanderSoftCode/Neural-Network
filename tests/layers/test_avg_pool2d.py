import numpy as np
import aether.config as config
from tests.base_case import AetherBaseTestCase

from aether.layers.conv import Pooling
TARGET_LAYER = Pooling

backends_to_test = ['numpy']
# GPU check & loading loop
try:
    import cupy as cp
    backends_to_test.append('cupy')
except (ImportError, Exception):
    pass  
def make_suite(backend_name, Layer_Class):
    class_name = f"Test_{Layer_Class.__name__}_{backend_name.upper()}"
    class TestPooling(AetherBaseTestCase):
        FILTER_SIZE = (2, 2)
        STRIDES = (2, 2)
        PADDING = 'valid'
        POOLING_TYPE = 'max'
        
        def setUp(self):
            # Function from aether.config.py 
            config.set_backend(backend_name=backend_name)
            self.xp = config.xp

            # Grab extra utility imports for specific backend if needed
            self.as_strided = config.get_stride_utility(self.xp)
            self.layer = Layer_Class(
                filter_size=self.FILTER_SIZE,
                stride=self.STRIDES,
                padding=self.PADDING,
                pooling_type=self.POOLING_TYPE
            )

            self.test_images = self.xp.random.randn(2, 28, 28, 1)

            # If a backend requires seperate functions, mimic Model.to
            # otherwise do nothing
            if hasattr(self.layer, '_compile_for_device'):
                self.layer._compile_for_device(backend_name)

        def test_forward_average_valid_shape(self):
            """Verify output dimensions for average pooling with 'valid' padding."""
            layer = Layer_Class(filter_size=(2, 2), stride=(2, 2),
                                 padding='valid', pooling_type='average')
            output = layer.forward(self.test_images, training=True)
            self.assertEqual(output.shape, (2, 14, 14, 1))

        def test_forward_average_same_shape(self):
            """Verify output dimensions for average pooling with 'same' padding
            on a non-evenly-divisible input."""
            layer = Layer_Class(filter_size=(2, 2), stride=(2, 2),
                                 padding='same', pooling_type='average')
            inputs = self.xp.random.randn(2, 7, 7, 3)
            output = layer.forward(inputs, training=True)
            self.assertEqual(output.shape, (2, 4, 4, 3))

        def test_forward_average_known_values(self):
            """Verify average pooling computes the correct mean of each window."""
            layer = Layer_Class(filter_size=(2, 2), stride=(2, 2),
                                 padding='valid', pooling_type='average')
            inputs = self.xp.arange(16, dtype=self.xp.float32).reshape(1, 4, 4, 1)
            expected = self.xp.array(
                [[[[2.5], [4.5]],
                  [[10.5], [12.5]]]], dtype=self.xp.float32
            )
            output = layer.forward(inputs, training=True)
            self.xp.testing.assert_array_almost_equal(output, expected, decimal=5)

        # ---- Backward pass Average Pooling Non-Overlapping Windows

        def test_backward_average_non_overlapping_valid(self):
            layer = Layer_Class(filter_size=(2, 2), stride=(2, 2),
                                 padding='valid', pooling_type='average')
            output = layer.forward(self.test_images, training=True)
            dvalues = self.xp.random.randn(*output.shape)
            dinputs = layer.backward(dvalues)
            self.assertEqual(dinputs.shape, self.test_images.shape)

        def test_backward_average_non_overlapping_same(self):
            layer = Layer_Class(filter_size=(2, 2), stride=(2, 2),
                                 padding='same', pooling_type='average')
            # Non-divisible spatial size so 'same' padding is non-trivial.
            inputs = self.xp.random.randn(2, 7, 7, 1)
            output = layer.forward(inputs, training=True)
            dvalues = self.xp.random.randn(*output.shape)
            dinputs = layer.backward(dvalues)
            self.assertEqual(dinputs.shape, inputs.shape)

        def test_backward_average_even_gradient_distribution(self):
            """Each element inside a non-overlapping average-pooling window should
            receive an equal share (dvalue / window_size) of the upstream gradient."""
            layer = Layer_Class(filter_size=(2, 2), stride=(2, 2),
                                 padding='valid', pooling_type='average')
            inputs = self.xp.arange(16, dtype=self.xp.float32).reshape(1, 4, 4, 1)
            layer.forward(inputs, training=True)

            dvalues = self.xp.array(
                [[[[4.], [8.]],
                  [[12.], [16.]]]], dtype=self.xp.float32
            )
            dinputs = layer.backward(dvalues)

            expected = self.xp.array(
                [[[[1.], [1.], [2.], [2.]],
                  [[1.], [1.], [2.], [2.]],
                  [[3.], [3.], [4.], [4.]],
                  [[3.], [3.], [4.], [4.]]]], dtype=self.xp.float32
            )
            self.xp.testing.assert_array_almost_equal(dinputs, expected, decimal=5)

        # ---- Backward pass Average Pooling Overlapping Windows

        def test_backward_average_overlapping_valid(self):
            layer = Layer_Class(filter_size=(3, 3), stride=(2, 2),
                                 padding='valid', pooling_type='average')
            output = layer.forward(self.test_images, training=True)
            dvalues = self.xp.random.randn(*output.shape)
            dinputs = layer.backward(dvalues)
            self.assertEqual(dinputs.shape, self.test_images.shape)

        def test_backward_average_overlapping_same(self):
            layer = Layer_Class(filter_size=(3, 3), stride=(2, 2),
                                 padding='same', pooling_type='average')
            output = layer.forward(self.test_images, training=True)
            dvalues = self.xp.random.randn(*output.shape)
            dinputs = layer.backward(dvalues)
            self.assertEqual(dinputs.shape, self.test_images.shape)

        def test_backward_average_overlapping_gradient_accumulation(self):
            """When average-pooling windows overlap, an input cell shared by
            multiple windows should accumulate a contribution from each one."""
            layer = Layer_Class(filter_size=(2, 2), stride=(1, 1),
                                 padding='valid', pooling_type='average')
            inputs = self.xp.random.randn(1, 3, 3, 1)
            layer.forward(inputs, training=True)

            dvalues = self.xp.ones((1, 2, 2, 1), dtype=self.xp.float32)
            dinputs = layer.backward(dvalues)

            expected = self.xp.array(
                [[[[0.25], [0.5], [0.25]],
                  [[0.5], [1.0], [0.5]],
                  [[0.25], [0.5], [0.25]]]], dtype=self.xp.float32
            )
            self.xp.testing.assert_array_almost_equal(dinputs, expected, decimal=5)
            # Gradient mass is conserved: each window's dvalue is fully
            # redistributed among its own cells, so summing every
            # (possibly overlapping) contribution must equal sum(dvalues).
            self.assertAlmostEqual(
                float(self.xp.sum(dinputs)), float(self.xp.sum(dvalues)), places=5
            )

        def test_backward_average_numerical_gradient(self):
            """Compare analytical gradient against 
            finite-difference approximation for average pooling."""
            epsilon = 1e-2
            layer = Layer_Class(filter_size=(2, 2), stride=(2, 2),
                                 padding='valid', pooling_type='average')

            fixed_input = self.xp.random.randn(2, 4, 4, 1)
            output = layer.forward(fixed_input, training=True)
            dvalues = self.xp.ones_like(output)

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

    TestPooling.__name__ = class_name
    TestPooling.__qualname__ = class_name
    return TestPooling

# Global Unpacking Loop
for backend in backends_to_test:
    
    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"
    # Bind generated class to the global namespace for unittest discovery
    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)