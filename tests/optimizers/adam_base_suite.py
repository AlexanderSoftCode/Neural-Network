import gc
import weakref

import aether.config as config
from tests.base_case import AetherBaseTestCase
from aether.layers.linear import Dense

backends_to_test = ['numpy']
try:
    import cupy as cp
    backends_to_test.append('cupy')
except (ImportError, Exception):
    pass


def make_suite(backend_name, Optimizer_Class):
    class TestOptimizerAdam(AetherBaseTestCase):
        LR = 0.001
        DECAY = 0.0
        EPSILON = 1e-7
        BETA_1 = 0.9
        BETA_2 = 0.999
        N_INPUTS = 4
        N_NEURONS = 5

        def setUp(self):
            self.backend_name = config.set_backend(backend_name=backend_name)
            self.xp = config.xp
            self.optimizer = self.make_layer(
                Optimizer_Class,
                learning_rate=self.LR,
                decay=self.DECAY,
                epsilon=self.EPSILON,
                beta_1=self.BETA_1,
                beta_2=self.BETA_2,
            )

            self.layer = self.make_layer(
                Dense, n_inputs=self.N_INPUTS, n_neurons=self.N_NEURONS
            )
            self.layer.dweights = self.xp.random.randn(*self.layer.weights.shape).astype(
                self.xp.float32
            )
            self.layer.dbiases = self.xp.random.randn(*self.layer.biases.shape).astype(
                self.xp.float32
            )

        # ---- state initialization ---------------------------------------

        def test_first_update_initializes_momentum_and_cache(self):
            """First call to update_parameters should lazily create weight/bias momentum + cache buffers."""
            self.assertFalse(hasattr(self.layer, "weight_momentums"))

            self.optimizer.update_parameters(self.layer)

            self.assertTrue(hasattr(self.layer, "weight_momentums"))
            self.assertTrue(hasattr(self.layer, "weight_cache"))
            self.assertTrue(hasattr(self.layer, "bias_momentums"))
            self.assertTrue(hasattr(self.layer, "bias_cache"))

            self.assertEqual(self.layer.weight_momentums.shape, self.layer.weights.shape)
            self.assertEqual(self.layer.weight_cache.shape, self.layer.weights.shape)
            self.assertEqual(self.layer.bias_momentums.shape, self.layer.biases.shape)
            self.assertEqual(self.layer.bias_cache.shape, self.layer.biases.shape)

        def test_second_update_accumulates_rather_than_resets(self):
            """Momentum/cache should carry state across steps."""
            self.optimizer.update_parameters(self.layer)
            momentums_after_first = self.layer.weight_momentums.copy()

            self.layer.dweights = self.xp.random.randn(*self.layer.weights.shape).astype(
                self.xp.float32
            )
            self.optimizer.update_parameters(self.layer)

            reset_value = (1 - self.optimizer.beta_1) * self.layer.dweights
            self.assertFalse(
                self.xp.allclose(self.layer.weight_momentums, reset_value),
                msg="weight_momentums looks reinitialized rather than accumulated.",
            )
            self.assertFalse(
                self.xp.allclose(momentums_after_first, self.layer.weight_momentums)
            )

        # ---- weakref-based layer tracking ---------------------------------

        def test_layer_tracking_does_not_prevent_garbage_collection(self):
            """Verify a dropped layer can still be garbage collected."""
            throwaway_layer = self.make_layer(Dense, n_inputs=2, n_neurons=2)
            throwaway_layer.dweights = self.xp.random.randn(
                *throwaway_layer.weights.shape
            ).astype(self.xp.float32)
            throwaway_layer.dbiases = self.xp.random.randn(
                *throwaway_layer.biases.shape
            ).astype(self.xp.float32)

            ref = weakref.ref(throwaway_layer)
            self.optimizer.update_parameters(throwaway_layer)
            self.assertIsNotNone(ref())

            del throwaway_layer
            gc.collect()

            self.assertIsNone(
                ref(),
                msg="Layer was not collected -- optimizer appears to hold a strong reference.",
            )

        # ---- bias correction correctness -----------------------------------

        def test_bias_corrections_formula(self):
            """_bias_corrections() formula check."""
            if not hasattr(self.optimizer, "_bias_corrections"):
                self.skipTest("Optimizer has no _bias_corrections() method.")

            self.optimizer.iterations = 3
            bc1, bc2 = self.optimizer._bias_corrections()

            expected_bc1 = 1 - self.optimizer.beta_1 ** (self.optimizer.iterations + 1)
            expected_bc2 = 1 - self.optimizer.beta_2 ** (self.optimizer.iterations + 1)

            self.assertAlmostEqual(bc1, expected_bc1, places=10)
            self.assertAlmostEqual(bc2, expected_bc2, places=10)

        # ---- learning rate decay -------------------------------------------

        def test_pre_update_applies_decay(self):
            decayed_optimizer = Optimizer_Class(learning_rate=0.1, decay=0.01)
            if hasattr(decayed_optimizer, "_compile_for_device"):
                decayed_optimizer._compile_for_device(backend_name)

            decayed_optimizer.iterations = 5
            decayed_optimizer.pre_update_parameters()

            expected_lr = 0.1 * (1.0 / (1.0 + 0.01 * 5))
            self.assertAlmostEqual(
                decayed_optimizer.current_learning_rate, expected_lr, places=7
            )

        def test_post_update_increments_iterations(self):
            start = self.optimizer.iterations
            self.optimizer.post_update_parameters()
            self.assertEqual(self.optimizer.iterations, start + 1)

        # ---- update math correctness -----------------------------------------

        def test_update_matches_manual_adam_step(self):
            """Single-step check against textbook Adam update."""
            weights_before = self.layer.weights.copy()
            biases_before = self.layer.biases.copy()
            dweights = self.layer.dweights.copy()
            dbiases = self.layer.dbiases.copy()

            beta_1 = self.optimizer.beta_1
            beta_2 = self.optimizer.beta_2
            eps = self.optimizer.epsilon
            lr = self.optimizer.current_learning_rate

            self.optimizer.update_parameters(self.layer)

            m_w = (1 - beta_1) * dweights
            v_w = (1 - beta_2) * (dweights**2)
            m_w_hat = m_w / (1 - beta_1**1)
            v_w_hat = v_w / (1 - beta_2**1)
            expected_weights = weights_before - lr * m_w_hat / (
                self.xp.sqrt(v_w_hat) + eps
            )

            m_b = (1 - beta_1) * dbiases
            v_b = (1 - beta_2) * (dbiases**2)
            m_b_hat = m_b / (1 - beta_1**1)
            v_b_hat = v_b / (1 - beta_2**1)
            expected_biases = biases_before - lr * m_b_hat / (
                self.xp.sqrt(v_b_hat) + eps
            )

            self.xp.testing.assert_array_almost_equal(
                self.layer.weights, expected_weights, decimal=5
            )
            self.xp.testing.assert_array_almost_equal(
                self.layer.biases, expected_biases, decimal=5
            )

        def test_multi_step_descent_direction(self):
            start_weights = self.layer.weights.copy()

            for _ in range(10):
                self.layer.dweights = self.xp.ones(
                    self.layer.weights.shape, dtype=self.xp.float32
                )
                self.layer.dbiases = self.xp.ones(
                    self.layer.biases.shape, dtype=self.xp.float32
                )
                self.optimizer.update_parameters(self.layer)
                self.optimizer.post_update_parameters()

            self.assertTrue(self.xp.all(self.layer.weights < start_weights))

        def test_update_produces_finite_values(self):
            self.optimizer.update_parameters(self.layer)
            self.assertTrue(self.xp.all(self.xp.isfinite(self.layer.weights)))
            self.assertTrue(self.xp.all(self.xp.isfinite(self.layer.biases)))

        # ---- in-place aliasing preservation ------------------------------------

        def test_update_preserves_weight_buffer_identity(self):
            weights_id_before = id(self.layer.weights)
            biases_id_before = id(self.layer.biases)

            self.optimizer.update_parameters(self.layer)

            self.assertEqual(weights_id_before, id(self.layer.weights))
            self.assertEqual(biases_id_before, id(self.layer.biases))

        # --- regression tests ---------------------------------------------------

        def test_gpu_kernel_compilation_and_dtypes(self):
            if backend_name != "cupy":
                self.skipTest("GPU kernel test is only applicable to CuPy backend.")

            for dtype in [self.xp.float32, self.xp.float64]:
                layer = Dense(n_inputs=2, n_neurons=2)
                layer.weights = layer.weights.astype(dtype)
                layer.biases = layer.biases.astype(dtype)
                layer.dweights = self.xp.ones(layer.weights.shape, dtype=dtype)
                layer.dbiases = self.xp.ones(layer.biases.shape, dtype=dtype)

                self.optimizer.weight_decay = 0.01

                try:
                    self.optimizer.update_parameters(layer)
                except Exception as e:
                    self.fail(
                        f"CuPy kernel failed to compile or run for dtype {dtype}: {e}"
                    )

                self.assertTrue(self.xp.all(self.xp.isfinite(layer.weights)))

    return TestOptimizerAdam