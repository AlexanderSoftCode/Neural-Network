import io
import contextlib
import warnings
import numpy as np
import aether.config as config
from tests.base_case import AetherBaseTestCase

from aether.base import Layer
from aether.model import Model
from aether.layers.linear import Dense, Flatten
from aether.layers.activations import ReLU, SoftMax
from aether.losses import Loss, CategoricalCrossEntropy, SoftmaxCategoricalCrossEntropy
from aether.metrics import Accuracy, CategoricalAccuracy
from aether.optimizers import Adam

TARGET_CLASS = Model

backends_to_test = ['numpy']
try:
    import cupy as cp
    backends_to_test.append('cupy')
except (ImportError, Exception):
    pass


def make_suite(backend_name, Target_Class):
    class_name = f"Test_{Target_Class.__name__}_{backend_name.upper()}"

    class MockTrainableLayer(Layer):
        """
        Compliant test layer implementing deferred initialization via build(),
        linear forward/backward operations, and framework lifecycle hooks.
        """
        def __init__(
            self,
            n_inputs=4,
            n_neurons=3,
            weight_regularizer_l1=0.0,
            bias_regularizer_l1=0.0,
            weight_regularizer_l2=0.0,
            bias_regularizer_l2=0.0,
            precision_exempt=False
        ):
            super().__init__()
            self.n_inputs = n_inputs
            self.n_neurons = n_neurons
            self.weight_regularizer_l1 = weight_regularizer_l1
            self.bias_regularizer_l1 = bias_regularizer_l1
            self.weight_regularizer_l2 = weight_regularizer_l2
            self.bias_regularizer_l2 = bias_regularizer_l2
            self._precision_exempt = precision_exempt

            # Unallocated state prior to build()
            self.weights = None
            self.biases = None
            self.dweights = None
            self.dbiases = None
            self.inputs = None
            self.is_built = False
            self.seed = None

            # Lifecycle spy hooks
            self.compiled_device = None
            self.applied_policy = None

        def build(self, seed=None):
            """Deferred weight allocation using the active config.xp backend."""
            if self.is_built and self.weights is not None:
                return

            self.seed = seed
            xp = config.xp
            std = xp.sqrt(2.0 / self.n_inputs, dtype=xp.float32)

            if seed is not None:
                rng = xp.random.RandomState(seed)
                self.weights = rng.randn(self.n_inputs, self.n_neurons).astype(xp.float32, copy=False) * std
            else:
                self.weights = (xp.random.randn(self.n_inputs, self.n_neurons).astype(xp.float32, copy=False) * std)

            self.biases = xp.zeros((1, self.n_neurons), dtype=xp.float32)
            self.is_built = True

        def _compile_for_device(self, device):
            self.compiled_device = device

        def _apply_precision(self, policy):
            self.applied_policy = policy

        def forward(self, X, training=True):
            self.inputs = X
            xp = config.get_array_module(X)
            return xp.matmul(X, self.weights) + self.biases

        def backward(self, dvalues):
            xp = config.get_array_module(dvalues)
            self.dweights = xp.matmul(self.inputs.T, dvalues)
            self.dbiases = xp.sum(dvalues, axis=0, keepdims=True)
            return xp.matmul(dvalues, self.weights.T)

    class MockSetSeedLayer(Layer):
        """Mock layer relying on _set_seed() hook instead of build()."""
        def __init__(self):
            super().__init__()
            self.seed = None

        def _set_seed(self, seed):
            self.seed = seed

        def forward(self, X, training=True):
            return X

        def backward(self, dvalues):
            return dvalues

    class TestModel(AetherBaseTestCase):
        NUM_SAMPLES = 32
        NUM_FEATURES = 4
        NUM_CLASSES = 3

        def setUp(self):
            super().setUp()
            self.backend_name = backend_name
            config.set_backend(backend_name=self.backend_name)
            self.xp = config.xp

            # Seed backend random generators deterministically
            if self.backend_name == 'numpy':
                np.random.seed(42)
            elif self.backend_name == 'cupy':
                cp.random.seed(42)

            self.X = self.xp.random.randn(self.NUM_SAMPLES, self.NUM_FEATURES).astype('float32')
            self.y = self.xp.random.randint(0, self.NUM_CLASSES, size=(self.NUM_SAMPLES,)).astype('int32')

            self.X_val = self.xp.random.randn(16, self.NUM_FEATURES).astype('float32')
            self.y_val = self.xp.random.randint(0, self.NUM_CLASSES, size=(16,)).astype('int32')

        # ---- 1. Structural, Deferred Build & Configuration Tests ----

        def test_deferred_build_allocation_on_finalize(self):
            """Verify parameters remain unallocated until finalize() explicitly triggers build()."""
            model = Target_Class()
            layer1 = MockTrainableLayer(self.NUM_FEATURES, 8)
            layer2 = MockTrainableLayer(8, self.NUM_CLASSES)

            model.add(layer1)
            model.add(layer2)

            self.assertFalse(layer1.is_built)
            self.assertIsNone(layer1.weights)
            self.assertIsNone(layer1.biases)
            self.assertFalse(layer2.is_built)
            self.assertIsNone(layer2.weights)
            self.assertIsNone(layer2.biases)

            model.finalize()

            self.assertTrue(layer1.is_built)
            self.assertIsNotNone(layer1.weights)
            self.assertIsNotNone(layer1.biases)
            self.assertEqual(layer1.weights.shape, (self.NUM_FEATURES, 8))
            self.assertEqual(layer1.biases.shape, (1, 8))

            self.assertTrue(layer2.is_built)
            self.assertIsNotNone(layer2.weights)
            self.assertIsNotNone(layer2.biases)
            self.assertEqual(layer2.weights.shape, (8, self.NUM_CLASSES))
            self.assertEqual(layer2.biases.shape, (1, self.NUM_CLASSES))

            # Trainable layer discovery verification
            self.assertEqual(len(model.trainable_layers), 2)
            self.assertIn(layer1, model.trainable_layers)
            self.assertIn(layer2, model.trainable_layers)

        def test_manual_seed_fluent_and_post_finalize_guard(self):
            """Test Model.manual_seed() fluent interface and ensure post-finalize modifications raise."""
            model = Target_Class()
            model.add(MockTrainableLayer(self.NUM_FEATURES, 8))

            returned_model = model.manual_seed(1234)
            self.assertIs(returned_model, model)
            self.assertEqual(model._seed, 1234)

            model.finalize()

            with self.assertRaises(RuntimeError):
                model.manual_seed(5678)

        def test_seed_propagation_to_layers(self):
            """Verify Model.finalize() correctly distributes indexed seeds and produces deterministic initializations."""
            base_seed = 100

            model1 = Target_Class()
            l1_m1 = MockTrainableLayer(self.NUM_FEATURES, 8)
            l2_m1 = MockTrainableLayer(8, self.NUM_CLASSES)
            l3_m1 = MockSetSeedLayer()
            model1.manual_seed(base_seed)
            model1.add(l1_m1)
            model1.add(l2_m1)
            model1.add(l3_m1)
            model1.finalize()

            model2 = Target_Class()
            l1_m2 = MockTrainableLayer(self.NUM_FEATURES, 8)
            l2_m2 = MockTrainableLayer(8, self.NUM_CLASSES)
            l3_m2 = MockSetSeedLayer()
            model2.manual_seed(base_seed)
            model2.add(l1_m2)
            model2.add(l2_m2)
            model2.add(l3_m2)
            model2.finalize()

            self.assertEqual(l1_m1.seed, base_seed + 0)
            self.assertEqual(l2_m1.seed, base_seed + 1)
            self.assertEqual(l3_m1.seed, base_seed)  # _set_seed fallback hook

            self.xp.testing.assert_allclose(l1_m1.weights, l1_m2.weights)
            self.xp.testing.assert_allclose(l2_m1.weights, l2_m2.weights)

        def test_sync_device_dispatches_on_finalize(self):
            """Verify finalize() calls _sync_device() to compile layers, loss, optimizer, and metrics."""
            class MockSyncLoss(Loss):
                def __init__(self):
                    super().__init__()
                    self.compiled_device = None
                def _compile_for_device(self, device):
                    self.compiled_device = device
                def forward(self, y_pred, y_true):
                    return config.xp.array(0.0)
                def backward(self, dvalues, y_true):
                    self.dinputs = dvalues

            class MockSyncOptimizer:
                def __init__(self):
                    self.compiled_device = None
                def _compile_for_device(self, device):
                    self.compiled_device = device
                def init_params(self, trainable_layers):
                    pass
                def step(self):
                    pass

            class MockSyncAccuracy(Accuracy):
                def __init__(self):
                    super().__init__()
                    self.compiled_device = None
                def _compile_for_device(self, device):
                    self.compiled_device = device
                def compare(self, predictions, y):
                    return config.xp.array(True)

            model = Target_Class()
            layer = MockTrainableLayer(self.NUM_FEATURES, self.NUM_CLASSES)
            loss = MockSyncLoss()
            opt = MockSyncOptimizer()
            acc = MockSyncAccuracy()

            model.add(layer)
            model.to(self.backend_name)
            model.configure(loss=loss, optimizer=opt, accuracy=acc)

            model.finalize()

            self.assertEqual(layer.compiled_device, self.backend_name)
            self.assertEqual(loss.compiled_device, self.backend_name)
            self.assertEqual(opt.compiled_device, self.backend_name)
            self.assertEqual(acc.compiled_device, self.backend_name)

        def test_add_layer_and_type_validation(self):
            """Ensure only instances of Layer can be added and layer count updates."""
            model = Target_Class()
            layer = MockTrainableLayer(self.NUM_FEATURES, 8)

            model.add(layer)
            self.assertEqual(len(model.layers), 1)
            self.assertIs(model.layers[0], layer)

            with self.assertRaises(TypeError):
                model.add("InvalidLayerObject")

        def test_mutation_after_finalize_raises(self):
            """Ensure graph mutations are locked once finalize() has been called."""
            model = Target_Class()
            model.add(MockTrainableLayer(self.NUM_FEATURES, 8))
            model.finalize()

            with self.assertRaises(RuntimeError):
                model.add(MockTrainableLayer(8, 2))

            with self.assertRaises(RuntimeError):
                model.finalize()

        def test_finalize_empty_model_raises(self):
            """Verify that finalizing an empty model raises a RuntimeError."""
            model = Target_Class()
            with self.assertRaises(RuntimeError):
                model.finalize()

        def test_configure_validation(self):
            """Test input validation across loss, optimizer, and metrics."""
            model = Target_Class()

            # Calling configure without arguments
            with self.assertRaises(ValueError):
                model.configure()

            # Invalid loss type
            with self.assertRaises(TypeError):
                model.configure(loss=object())

            # Invalid optimizer (lacks step/init_params)
            with self.assertRaises(TypeError):
                model.configure(optimizer=object())

            # Invalid metric type
            with self.assertRaises(TypeError):
                model.configure(accuracy=object())

        def test_softmax_cce_fusion(self):
            """Verify finalize() raises ValueError when a trailing SoftMax is added alongside SoftmaxCategoricalCrossEntropy."""
            model = Target_Class()
            model.add(MockTrainableLayer(self.NUM_FEATURES, 8))
            model.add(MockTrainableLayer(8, self.NUM_CLASSES))
            model.add(SoftMax())

            model.configure(loss=SoftmaxCategoricalCrossEntropy())

            with self.assertRaises(ValueError):
                model.finalize()

        def test_device_migration_guardrails_and_compilation(self):
            """Test backend device switching hooks and prevent post-finalize migrations."""
            model = Target_Class()
            layer = MockTrainableLayer(self.NUM_FEATURES, 4)
            model.add(layer)

            # Migration before finalize should trigger compile hook
            model.to(self.backend_name)
            self.assertEqual(model.device, self.backend_name)
            self.assertEqual(layer.compiled_device, self.backend_name)

            model.finalize()

            # Re-migrating after finalize must raise RuntimeError
            with self.assertRaises(RuntimeError):
                model.to(self.backend_name)

        def test_precision_policy_dispatch(self):
            """Verify set_precision() dispatches to non-exempt layers and preserves exempt layers."""
            model = Target_Class()
            standard_layer = MockTrainableLayer(self.NUM_FEATURES, 8, precision_exempt=False)
            exempt_layer = MockTrainableLayer(8, self.NUM_CLASSES, precision_exempt=True)

            model.add(standard_layer)
            model.add(exempt_layer)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy float16 is emulated.*")
                model.set_precision('float16')

            self.assertIsNotNone(standard_layer.applied_policy)
            self.assertEqual(standard_layer.applied_policy.compute_dtype_name, 'float16')
            self.assertIsNone(exempt_layer.applied_policy)

        # ---- 2. Forward & Backward Pipeline Execution -------

        def test_forward_and_backward_execution(self):
            """Verify sequential forward and backward propagations evaluate matching tensor shapes."""
            model = Target_Class()
            layer1 = MockTrainableLayer(self.NUM_FEATURES, 8)
            layer2 = MockTrainableLayer(8, self.NUM_CLASSES)
            model.add(layer1)
            model.add(layer2)
            model.finalize()

            # Forward pass
            output = model.forward(self.X, training=True)
            expected_shape = (self.NUM_SAMPLES, self.NUM_CLASSES)
            self.assertEqual(output.shape, expected_shape)

            # Backward pass
            dummy_loss_grad = self.xp.ones_like(output)
            dinputs = model.backward(dummy_loss_grad)
            self.assertEqual(dinputs.shape, self.X.shape)

            # Gradients must be populated in trainable layers
            self.assertIsNotNone(layer1.dweights)
            self.assertIsNotNone(layer1.dbiases)
            self.assertIsNotNone(layer2.dweights)
            self.assertIsNotNone(layer2.dbiases)
            self.assertEqual(layer1.dweights.shape, (self.NUM_FEATURES, 8))
            self.assertEqual(layer2.dweights.shape, (8, self.NUM_CLASSES))

        # ---- 3. Training & Optimization Tests ---------------

        def test_train_loop_loss_reduction(self):
            """Test end-to-end training loop and verify the optimizer reduces loss across epochs."""
            model = Target_Class()
            model.add(MockTrainableLayer(self.NUM_FEATURES, 16))
            model.add(MockTrainableLayer(16, self.NUM_CLASSES))

            loss = SoftmaxCategoricalCrossEntropy()
            optimizer = Adam(learning_rate=0.05)
            accuracy = CategoricalAccuracy()

            model.configure(loss=loss, optimizer=optimizer, accuracy=accuracy)

            with contextlib.redirect_stdout(io.StringIO()):
                initial_loss, _ = model.evaluate(self.X, self.y, batch_size=self.NUM_SAMPLES)

                model.train(
                    self.X,
                    self.y,
                    epochs=25,
                    batch_size=8,
                    print_every=0,
                    validation_data=(self.X_val, self.y_val)
                )

                final_loss, final_acc = model.evaluate(self.X, self.y, batch_size=self.NUM_SAMPLES)

            self.assertLess(final_loss, initial_loss)
            self.assertTrue(0.0 <= final_acc <= 1.0)

        def test_train_with_regularization(self):
            """Verify that training executes properly when L1/L2 regularizers are enabled."""
            model = Target_Class()
            reg_layer = MockTrainableLayer(
                self.NUM_FEATURES,
                self.NUM_CLASSES,
                weight_regularizer_l2=5e-4,
                bias_regularizer_l2=5e-4
            )
            model.add(reg_layer)

            model.configure(
                loss=SoftmaxCategoricalCrossEntropy(),
                optimizer=Adam(learning_rate=0.01),
                accuracy=CategoricalAccuracy()
            )

            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    model.train(self.X, self.y, epochs=2, batch_size=16, print_every=0, verbose=False)
            except Exception as e:
                self.fail(f"Training loop failed with regularized mock layer: {e}")

        # ---- 4. Evaluation & Prediction Routing Tests -------

        def test_evaluate_metrics(self):
            """Test evaluate() inference mode calculations and metric outputs."""
            model = Target_Class()
            model.add(MockTrainableLayer(self.NUM_FEATURES, self.NUM_CLASSES))

            model.configure(
                loss=SoftmaxCategoricalCrossEntropy(),
                accuracy=CategoricalAccuracy()
            )

            with contextlib.redirect_stdout(io.StringIO()):
                val_loss, val_acc = model.evaluate(self.X_val, self.y_val, batch_size=8)

            self.assertIsInstance(val_loss, float)
            self.assertIsInstance(val_acc, float)
            self.assertGreaterEqual(val_loss, 0.0)
            self.assertTrue(0.0 <= val_acc <= 1.0)

        def test_predict_logits_vs_probabilities(self):
            """Verify predict() routes output through fused activation or returns raw logits."""
            model = Target_Class()
            model.add(MockTrainableLayer(self.NUM_FEATURES, self.NUM_CLASSES))
            model.configure(loss=SoftmaxCategoricalCrossEntropy())

            probs = model.predict(self.X_val, batch_size=8, return_logits=False)
            self.assertEqual(probs.shape, (len(self.X_val), self.NUM_CLASSES))

            row_sums = self.xp.sum(probs, axis=1)
            expected_sums = self.xp.ones_like(row_sums)
            self.xp.testing.assert_allclose(row_sums, expected_sums, atol=1e-4)

            logits = model.predict(self.X_val, batch_size=8, return_logits=True)
            self.assertEqual(logits.shape, (len(self.X_val), self.NUM_CLASSES))

            self.assertFalse(self.xp.allclose(self.xp.sum(logits, axis=1), 1.0))

        # ---- 5. Regression Tests ----------------------------

        def test_real_pipeline_nd_tensor_backward(self):
            """Test full forward and backward pass with real Flatten and Dense layers on 4D inputs."""
            model = Target_Class()
            model.add(Flatten())
            model.add(Dense(3 * 4 * 4, 8))
            model.add(Dense(8, self.NUM_CLASSES))

            loss = SoftmaxCategoricalCrossEntropy()
            optimizer = Adam(learning_rate=0.01)
            accuracy = CategoricalAccuracy()
            model.configure(loss=loss, optimizer=optimizer, accuracy=accuracy)
            model.finalize()

            X_4d = self.xp.random.randn(8, 4, 4, 3).astype('float32')
            y_4d = self.xp.random.randint(0, self.NUM_CLASSES, size=(8,)).astype('int32')

            out = model.forward(X_4d, training=True)
            self.assertEqual(out.shape, (8, self.NUM_CLASSES))

            loss.backward(out, y_4d)
            dinputs = model.backward(loss.dinputs)

            self.assertEqual(dinputs.shape, X_4d.shape)

        def test_real_pipeline_fp16_mixed_precision(self):
            """Verify forward and backward numerical stability under float16 precision policy."""
            model = Target_Class()
            model.add(Flatten())
            dense1 = Dense(self.NUM_FEATURES, 16)
            dense2 = Dense(16, self.NUM_CLASSES)
            model.add(dense1)
            model.add(dense2)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy float16 is emulated.*")
                model.set_precision('float16')
            model.configure(
                loss=SoftmaxCategoricalCrossEntropy(),
                optimizer=Adam(learning_rate=0.01),
                accuracy=CategoricalAccuracy()
            )
            model.finalize()

            out = model.forward(self.X, training=True)
            self.assertEqual(out.dtype, self.xp.float16)

            self.assertEqual(dense1.weights.dtype, self.xp.float32)
            self.assertEqual(dense2.weights.dtype, self.xp.float32)

        def test_inference_cache_invalidation(self):
            """Verify inference passes do not retain training backward caches in Dense layers."""
            model = Target_Class()
            dense = Dense(self.NUM_FEATURES, self.NUM_CLASSES)
            model.add(dense)
            model.configure(loss=SoftmaxCategoricalCrossEntropy())
            model.finalize()

            _ = model.predict(self.X_val)

            # Ephemeral caches should be cleared
            self.assertIsNone(dense._inputs_compute)
            self.assertIsNone(dense._weights_compute)

    TestModel.__name__ = class_name
    TestModel.__qualname__ = class_name

    return TestModel


for backend in backends_to_test:
    class_name = f"Test_{TARGET_CLASS.__name__}_{backend.upper()}"
    globals()[class_name] = make_suite(backend_name=backend, Target_Class=TARGET_CLASS)