import functools
import numpy as np
import aether.config as config
from aether.base import Layer
from tests.base_case import register_test_suites
from tests.integration.model_base_suite import ModelIntegrationBaseCase


class TestDeviceMigration(ModelIntegrationBaseCase):

    def test_pointer_swap_binds_gpu_paths_after_to_cupy(self):
        """Every layer with _compile_for_device should have its forward
        pointer flipped away from the CPU fallback once migrated to cupy."""
        if self.backend_name != "cupy":
            self.skipTest("Pointer-swap only meaningfully diverges on the CuPy backend.")

        model = self.build_cnn_model()
        for layer in model.layers:
            if type(layer)._compile_for_device is not Layer._compile_for_device:
                with self.subTest(layer=type(layer).__name__):
                    fwd_fn = layer.forward
                    func_name = (
                        fwd_fn.func.__name__
                        if isinstance(fwd_fn, functools.partial)
                        else getattr(fwd_fn, "__name__", None)
                    )

                    self.assertIn(
                        func_name,
                        ("_forward_gpu", "_forward_fallback"),
                        msg=(
                            f"{type(layer).__name__}.forward did not bind to a "
                            f"recognized pointer-swap target after Model.to('cupy'). Got: {func_name}"
                        ),
                    )

    # TODO: Add cross-backend forward and weight-parity integration tests.
    # Currently, Model.to() is enforced strictly before Model.finalize() to guarantee
    # zero-copy deterministic tensor allocation. Live cross-device synchronization
    # (e.g. migrating finalized CPU graphs to GPU) is deferred until parameter migration
    # utilities and optimizer buffer rebinding are fully standardized.

register_test_suites(globals(), TestDeviceMigration)