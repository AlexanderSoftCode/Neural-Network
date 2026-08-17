import functools
import numpy as np
import aether as ae
import aether.config as config
from aether.base import Layer
from tests.integration.model_base_suite import ModelIntegrationBaseCase, backends_to_test

try:
    import cupy as cp
except (ImportError, Exception):
    cp = None


def make_suite(backend_name):
    class_name = f"Test_DeviceMigration_{backend_name.upper()}"

    class TestDeviceMigration(ModelIntegrationBaseCase):

        def setUp(self):
            self.backend_name = backend_name
            config.set_backend(backend_name=backend_name)
            self.xp = config.xp

        def test_pointer_swap_binds_gpu_paths_after_to_cupy(self):
            """Every layer with _compile_for_device should have its forward
            pointer flipped away from the CPU fallback once migrated to cupy,
            or fall back cleanly if the specific GPU kernel isn't available on
            this hardware -- either way forward pointer should resolve to
            _forward_gpu or _forward_fallback."""
            if self.backend_name != 'cupy':
                self.skipTest("Pointer-swap only meaningfully diverges on the CuPy backend.")

            model = self.build_cnn_model(device='cupy')
            for layer in model.layers:
                # Only test layers that explicitly override the base hook
                if type(layer)._compile_for_device is not Layer._compile_for_device:
                    with self.subTest(layer=type(layer).__name__):
                        # Unpack underlying function name if wrapped in functools.partial
                        fwd_fn = layer.forward
                        func_name = fwd_fn.func.__name__ if isinstance(fwd_fn, functools.partial) else getattr(fwd_fn, "__name__", None)

                        self.assertIn(
                            func_name,
                            ("_forward_gpu", "_forward_fallback"),
                            msg=f"{type(layer).__name__}.forward did not bind to a "
                                f"recognized pointer-swap target after Model.to('cupy'). Got: {func_name}"
                        )

        def test_forward_shapes_consistent_across_backends(self):
            """Independent random inits per backend (numpy vs cupy RNG streams
            differ), so this only checks shape/finiteness parity, not exact
            values -- exact-value parity is covered separately below."""
            if not config.HAS_CUPY:
                self.skipTest("CuPy not available in this environment.")

            config.set_backend('numpy')
            np_model = self.build_mlp_model(device='numpy')
            X_np, _ = self.make_synthetic_image_data(n_samples=4)
            np_out = np_model.forward(X_np, training=False)

            config.set_backend('cupy')
            gpu_model = self.build_mlp_model(device='cupy')
            X_gpu = cp.asarray(np.asarray(X_np))
            gpu_out = gpu_model.forward(X_gpu, training=False)

            self.assertEqual(np_out.shape, gpu_out.shape)
            self.assertTrue(bool(np.all(np.isfinite(np_out))))
            self.assertTrue(bool(cp.all(cp.isfinite(gpu_out))))

        def test_cpu_gpu_forward_parity_after_weight_sync(self):
            """Build the same MLP on numpy and cupy, force the GPU copy's
            weights/biases to match the CPU copy exactly, then confirm the two
            forward passes agree numerically. This is the strongest device
            parity check -- unlike test_forward_shapes_consistent_across_backends
            it can't pass just because both sides happen to produce finite
            output of the right shape."""
            if not config.HAS_CUPY:
                self.skipTest("CuPy not available in this environment.")

            config.set_backend('numpy')
            cpu_model = self.build_mlp_model(device='numpy')
            X_np, _ = self.make_synthetic_image_data(n_samples=4)
            cpu_out = cpu_model.forward(X_np, training=False)

            config.set_backend('cupy')
            gpu_model = self.build_mlp_model(device='cupy')

            self.assertEqual(len(cpu_model.trainable_layers), len(gpu_model.trainable_layers))
            for cpu_layer, gpu_layer in zip(cpu_model.trainable_layers, gpu_model.trainable_layers):
                gpu_layer.weights = cp.asarray(np.asarray(cpu_layer.weights))
                gpu_layer.biases = cp.asarray(np.asarray(cpu_layer.biases))

            X_gpu = cp.asarray(np.asarray(X_np))
            gpu_out = gpu_model.forward(X_gpu, training=False)

            np.testing.assert_allclose(cp.asnumpy(gpu_out), np.asarray(cpu_out), rtol=1e-4, atol=1e-4)

    TestDeviceMigration.__name__ = class_name
    TestDeviceMigration.__qualname__ = class_name
    return TestDeviceMigration

for backend in backends_to_test:
    globals()[f"Test_DeviceMigration_{backend.upper()}"] = make_suite(backend)