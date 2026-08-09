import numpy as np
import aether.config as config
from tests.base_case import AetherBaseTestCase

from aether.layers.pooling import GlobalAvgPool
TARGET_LAYER = GlobalAvgPool

backends_to_test = ['numpy']
try:
    import cupy as cp
    backends_to_test.append('cupy')
except (ImportError, Exception):
    pass

def make_suite(backend_name, Layer_Class):
    class_name = f"Test_{Layer_Class.__name__}_{backend_name.upper()}"

    class TestGlobalAvgPool(AetherBaseTestCase):
        S, H, W, C = 2, 28, 28, 3
        def setUp(self):
            self.backend_name = backend_name
            config.set_backend(backend_name=self.backend_name)
            self.xp = config.xp

            self.layer = self.make_layer(Layer_Class)
            self.test_images = self.xp.random.randn(self.S, self.H, self.W, self.C).astype(self.xp.float32)
            func_obj = getattr(self.layer.forward, 'func', self.layer.forward)
            self.uses_gpu_kernel = (getattr(func_obj, '__name__', '') == '_forward_gpu')
        def test_gap_forward_shape(self):
            """Verify output dimensions based on samples and channels"""

            output = self.layer.forward(self.test_images, training=True)
            expected_shape = (self.S, self.C)
            self.assertEqual(output.shape, expected_shape)
        def test_forward_gpu_reduction_kernel_matches_cpu(self):
            
            if self.backend_name != 'cupy' or not config.HAS_CUPY:
                self.skipTest("Applies only to CuPy pass")


            layer_cpu = GlobalAvgPool()
            layer_cpu._compile_for_device("numpy")
            cpu_output = layer_cpu.forward(self.test_images, training=True)

            gpu_inputs = cp.array(self.test_images)
            layer_gpu = GlobalAvgPool()
            layer_gpu._compile_for_device("cupy")
            gpu_output = layer_gpu.forward(gpu_inputs, training=True)

            self.assertIsInstance(gpu_output, cp.ndarray)
            self.assertEqual(gpu_output.shape, (self.S, self.C))

            self.xp.testing.assert_allclose(
                cp.asnumpy(gpu_output),
                cpu_output,
                rtol=1e-5,
                atol=1e-5
            )
        def test_forward_known_values(self):
            """Verify mathematical output with uniform constant inputs."""
            constant_val = 5.5
            constant_input = self.xp.full((self.S, self.H, self.W, self.C), constant_val, dtype=self.xp.float32)
            
            out = self.layer.forward(constant_input, training=False)
            expected = self.xp.full((self.S, self.C), constant_val, dtype=self.xp.float32)
            
            self.xp.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

        def test_backward_gpu_kernel_matches_cpu_broadcast(self):
            """Cross-validate GPU backward kernel against CPU broadcasting fallback."""
            self.layer.forward(self.test_images, training=True)
            dvalues = self.xp.random.randn(self.S, self.C).astype(self.xp.float32)
            
            dinputs_actual = self.layer.backward(dvalues)
            
            inv_area = np.float32(1.0 / (self.H * self.W))
            dvalues4d = dvalues[:, self.xp.newaxis, self.xp.newaxis, :]
            dinputs_expected = self.xp.broadcast_to(dvalues4d, (self.S, self.H, self.W, self.C)) * inv_area
            
            self.xp.testing.assert_allclose(dinputs_actual, dinputs_expected, rtol=1e-5, atol=1e-6)

        def test_backward_gradient_sum_conservation(self):
            """Verify gradient conservation: sum(dinputs) per sample/channel equals dvalues."""
            self.layer.forward(self.test_images, training=True)
            dvalues = self.xp.random.randn(self.S, self.C).astype(self.xp.float32)
            
            dinputs = self.layer.backward(dvalues)
            
            # Summing dinputs across spatial axes (H, W) should yield back dvalues
            dinputs_sum = self.xp.sum(dinputs, axis=(1, 2))
            self.xp.testing.assert_allclose(dinputs_sum, dvalues, rtol=1e-5, atol=1e-6)

        def test_numerical_gradient_check(self):
            """Verify backward analytical gradient via centered finite differences."""
            S_small, H_small, W_small, C_small = 2, 4, 4, 2
            x_np = np.random.randn(S_small, H_small, W_small, C_small).astype(np.float32)
            x_xp = self.xp.array(x_np)
            
            small_layer = self.make_layer(Layer_Class)
            small_layer.forward(x_xp, training=True)
            
            dvalues_np = np.random.randn(S_small, C_small).astype(np.float32)
            dvalues_xp = self.xp.array(dvalues_np)
            
            analytical_dinputs = small_layer.backward(dvalues_xp)
            if hasattr(analytical_dinputs, "get"):
                analytical_dinputs = analytical_dinputs.get()

            eps = 1e-3
            numerical_dinputs = np.zeros_like(x_np)

            # Compute numerical gradient for each input scalar element
            it = np.nditer(x_np, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                
                # x + eps
                x_pos = x_np.copy()
                x_pos[idx] += eps
                out_pos = small_layer.forward(self.xp.array(x_pos), training=False)
                if hasattr(out_pos, "get"): out_pos = out_pos.get()
                
                # x - eps
                x_neg = x_np.copy()
                x_neg[idx] -= eps
                out_neg = small_layer.forward(self.xp.array(x_neg), training=False)
                if hasattr(out_neg, "get"): out_neg = out_neg.get()
                
                # Scalar dot product with dvalues
                loss_pos = np.sum(out_pos * dvalues_np)
                loss_neg = np.sum(out_neg * dvalues_np)
                
                numerical_dinputs[idx] = (loss_pos - loss_neg) / (2.0 * eps)
                it.iternext()

            np.testing.assert_allclose(analytical_dinputs, numerical_dinputs, rtol=1e-3, atol=1e-4)

        def test_shape_meta_launch_cache(self):
            """Verify grid/block launch metadata caching for custom GPU kernels."""
            if not self.uses_gpu_kernel:
                self.skipTest("Skipping GPU launch metadata test for CPU backend.")
                
            shape = (4, 14, 14, 64)
            meta_1 = self.layer._get_shape_meta(shape, self.layer._block_z)
            meta_2 = self.layer._get_shape_meta(shape, self.layer._block_z)
            
            self.assertIs(meta_1, meta_2)
            self.assertEqual(meta_1["out_shape"], (4, 64))

        def test_thin_channels_launch_geometry(self):
            """Verify launch geometry calculation when channels C <= 32."""
            thin_shape = (2, 28, 28, 8)
            meta = self.layer._get_shape_meta(thin_shape, block_z=4)
            
            self.assertEqual(meta["block_dim"][0], 8)
            self.assertEqual(meta["grid_dim"][0], 1)  # (8 + 8 - 1) // 8 == 1
            
    TestGlobalAvgPool.__name__ = class_name
    TestGlobalAvgPool.__qualname__ = class_name

    return TestGlobalAvgPool

for backend in backends_to_test:

    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"
    globals()[class_name] = make_suite(backend_name=backend, Layer_Class=TARGET_LAYER)