import json
import os
import shutil
import tempfile
import warnings
import zipfile
import numpy as np

import aether.config as config
from aether.layers.activations import LeakyReLU, ReLU
from aether.layers.conv import Conv2d
from aether.layers.linear import Dense, Flatten
from aether.layers.normalization import BatchNorm
from aether.layers.pooling import MaxPool2d
from aether.model import Model
from tests.base_case import AetherBaseTestCase

try:
    import safetensors.numpy

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

TARGET_CLASS = Model

backends_to_test = ["numpy"]
try:
    import cupy as cp

    backends_to_test.append("cupy")
except (ImportError, Exception):
    pass


def make_suite(backend_name, Target_Class):
    class_name = f"Test_{Target_Class.__name__}_Load_{backend_name.upper()}"

    class TestModelLoad(AetherBaseTestCase):
        NUM_FEATURES = 8
        NUM_CLASSES = 3

        def setUp(self):
            super().setUp()
            if not HAS_SAFETENSORS:
                self.skipTest(
                    "safetensors library is not installed in the environment."
                )

            self.backend_name = backend_name
            config.set_backend(backend_name=self.backend_name)
            self.xp = config.xp

            if self.backend_name == "numpy":
                np.random.seed(42)
            elif self.backend_name == "cupy":
                cp.random.seed(42)

            self.temp_dir = tempfile.mkdtemp()
            self.save_path = os.path.join(self.temp_dir, "test_model.aether")

        def tearDown(self):
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            super().tearDown()

        # ---- 1. Error Handling & Malformed Archive Tests ----

        def test_load_nonexistent_file_raises(self):
            """Ensure attempting to load a nonexistent path raises FileNotFoundError or OSError."""
            non_existent_path = os.path.join(self.temp_dir, "ghost_model.aether")
            with self.assertRaises((FileNotFoundError, OSError)):
                Target_Class.load(non_existent_path)

        def test_load_corrupted_archive_raises(self):
            """Ensure loading a corrupted/truncated zip archive raises BadZipFile."""
            corrupt_path = os.path.join(self.temp_dir, "corrupt_model.aether")
            with open(corrupt_path, "wb") as f:
                f.write(b"NOT_A_VALID_ZIP_PAYLOAD_DATA")

            with self.assertRaises(zipfile.BadZipFile):
                Target_Class.load(corrupt_path)

        def test_load_unknown_layer_class_raises(self):
            """Ensure manifest referencing an unresolvable layer class raises ValueError."""
            # Build and save a valid baseline model
            model = Target_Class()
            model.add(Dense(self.NUM_FEATURES, 16))
            model.finalize((self.NUM_FEATURES,))
            model.save(self.save_path)

            # Corrupt the architecture manifest inside the zip archive
            with zipfile.ZipFile(self.save_path, "r") as zipf:
                manifest = json.loads(zipf.read("architecture.json").decode("utf-8"))
                weights_bytes = zipf.read("weights.safetensors")

            manifest["layers"][0]["class_name"] = "HypotheticalAlienLayer"
            corrupt_manifest_bytes = json.dumps(manifest).encode("utf-8")

            corrupt_save_path = os.path.join(self.temp_dir, "corrupt_layer.aether")
            with zipfile.ZipFile(corrupt_save_path, "w") as zipf:
                zipf.writestr("architecture.json", corrupt_manifest_bytes, compress_type=zipfile.ZIP_DEFLATED)
                zipf.writestr("weights.safetensors", weights_bytes, compress_type=zipfile.ZIP_STORED)

            with self.assertRaises(ValueError) as ctx:
                Target_Class.load(corrupt_save_path)
            self.assertIn("Unknown layer class 'HypotheticalAlienLayer'", str(ctx.exception))

        # ---- 2. Basic Architecture & Parameter Hydration Tests ----

        def test_basic_mlp_graph_reconstruction_and_parameter_parity(self):
            """Verify reconstructed MLP matches original layer classes, configs, and exact tensor weights."""
            orig_model = Target_Class()
            orig_model.manual_seed(123)
            d1 = Dense(self.NUM_FEATURES, 16, l2 = 1e-4)
            act = ReLU()
            d2 = Dense(16, self.NUM_CLASSES)

            orig_model.add(d1)
            orig_model.add(act)
            orig_model.add(d2)
            orig_model.finalize((self.NUM_FEATURES,))
            orig_model.save(self.save_path)

            loaded_model = Target_Class.load(self.save_path)

            # Verify graph state and structure
            self.assertTrue(loaded_model.is_finalized)
            self.assertEqual(len(loaded_model.layers), 3)
            self.assertEqual(type(loaded_model.layers[0]), Dense)
            self.assertEqual(type(loaded_model.layers[1]), ReLU)
            self.assertEqual(type(loaded_model.layers[2]), Dense)

            # Verify layer configs
            self.assertEqual(loaded_model.layers[0].n_inputs, self.NUM_FEATURES)
            self.assertEqual(loaded_model.layers[0].n_neurons, 16)
            self.assertEqual(loaded_model.layers[0].weight_regularizer_l2, 1e-4)

            # Verify exact numerical parameter hydration
            np.testing.assert_allclose(
                config.to_device(loaded_model.layers[0].weights, target="numpy"),
                config.to_device(d1.weights, target="numpy"),
            )
            np.testing.assert_allclose(
                config.to_device(loaded_model.layers[0].biases, target="numpy"),
                config.to_device(d1.biases, target="numpy"),
            )
            np.testing.assert_allclose(
                config.to_device(loaded_model.layers[2].weights, target="numpy"),
                config.to_device(d2.weights, target="numpy"),
            )
            np.testing.assert_allclose(
                config.to_device(loaded_model.layers[2].biases, target="numpy"),
                config.to_device(d2.biases, target="numpy"),
            )

        # ---- 3. Inference & Prediction Parity Tests ----

        def test_forward_pass_and_prediction_numerical_parity(self):
            """Verify predictions of loaded model are identical to the source model in evaluation mode."""
            orig_model = Target_Class()
            orig_model.manual_seed(99)
            orig_model.add(Dense(self.NUM_FEATURES, 32))
            orig_model.add(LeakyReLU(alpha=0.2))
            orig_model.add(Dense(32, self.NUM_CLASSES))
            orig_model.finalize((self.NUM_FEATURES,))
            orig_model.save(self.save_path)

            loaded_model = Target_Class.load(self.save_path)

            # Generate synthetic test batch
            raw_X = np.random.randn(10, self.NUM_FEATURES).astype(np.float32)
            X = config.to_device(raw_X, target=self.backend_name)

            orig_preds = orig_model.predict(X)
            loaded_preds = loaded_model.predict(X)

            np.testing.assert_allclose(
                config.to_device(loaded_preds, target="numpy"),
                config.to_device(orig_preds, target="numpy"),
                rtol=1e-5,
                atol=1e-6,
            )

        # ---- 4. Precision Policy Restoration Tests ----

        def test_precision_policy_restoration(self):
            """Assert target compute dtype policy persists through serialization and re-hydrates properly."""
            orig_model = Target_Class()
            orig_model.manual_seed(42)
            orig_model.add(Dense(self.NUM_FEATURES, 16))
            orig_model.add(ReLU())
            orig_model.add(Dense(16, self.NUM_CLASSES))

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                orig_model.set_precision("float32")

            orig_model.finalize((self.NUM_FEATURES,))
            orig_model.save(self.save_path)

            loaded_model = Target_Class.load(self.save_path)

            self.assertIsNotNone(loaded_model.precision_policy)
            self.assertEqual(loaded_model.precision_policy.compute_dtype_name, "float32")
            self.assertEqual(loaded_model.layers[0].weights.dtype, np.float32)
            self.assertEqual(loaded_model.layers[2].weights.dtype, np.float32)

        # ---- 5. Complex CNN Architecture & Running Buffers ----

        def test_complex_cnn_architecture_and_batchnorm_buffers_load(self):
            """Test loading of a full CNN graph and verify persistent buffers (BatchNorm running stats)."""
            orig_model = Target_Class()
            orig_model.manual_seed(77)

            conv = Conv2d(
                in_channels=3,
                out_channels=8,
                filter_size=(3, 3),
                stride=(1, 1),
                padding="same",
            )
            bn = BatchNorm()
            pool = MaxPool2d(filter_size=(2, 2), stride=(2, 2))
            flatten = Flatten()
            dense = Dense(4 * 4 * 8, self.NUM_CLASSES)

            orig_model.add(conv)
            orig_model.add(bn)
            orig_model.add(ReLU())
            orig_model.add(pool)
            orig_model.add(flatten)
            orig_model.add(dense)
            orig_model.finalize((8, 8, 3))

            # Simulate training forward pass to update BatchNorm running statistics
            raw_input = np.random.randn(4, 8, 8, 3).astype(np.float32)
            X_sample = config.to_device(raw_input, target=self.backend_name)
            orig_model.forward(X_sample, training=True)

            orig_model.save(self.save_path)
            loaded_model = Target_Class.load(self.save_path)

            # Validate graph layer pipeline
            self.assertEqual(len(loaded_model.layers), 6)
            self.assertIsInstance(loaded_model.layers[0], Conv2d)
            self.assertIsInstance(loaded_model.layers[1], BatchNorm)
            self.assertIsInstance(loaded_model.layers[3], MaxPool2d)
            self.assertIsInstance(loaded_model.layers[4], Flatten)
            self.assertIsInstance(loaded_model.layers[5], Dense)

            # Verify Conv weights & biases
            np.testing.assert_allclose(
                config.to_device(loaded_model.layers[0].weights, target="numpy"),
                config.to_device(conv.weights, target="numpy"),
            )
            np.testing.assert_allclose(
                config.to_device(loaded_model.layers[0].biases, target="numpy"),
                config.to_device(conv.biases, target="numpy"),
            )

            # Verify BatchNorm learnable parameters and running mean/variance buffers
            loaded_bn = loaded_model.layers[1]
            np.testing.assert_allclose(
                config.to_device(loaded_bn.gamma, target="numpy"),
                config.to_device(bn.gamma, target="numpy"),
            )
            np.testing.assert_allclose(
                config.to_device(loaded_bn.beta, target="numpy"),
                config.to_device(bn.beta, target="numpy"),
            )
            np.testing.assert_allclose(
                config.to_device(loaded_bn.running_mean, target="numpy"),
                config.to_device(bn.running_mean, target="numpy"),
            )
            np.testing.assert_allclose(
                config.to_device(loaded_bn.running_var, target="numpy"),
                config.to_device(bn.running_var, target="numpy"),
            )

            # Verify end-to-end evaluation forward pass parity
            orig_eval = orig_model.forward(X_sample, training=False)
            loaded_eval = loaded_model.forward(X_sample, training=False)

            np.testing.assert_allclose(
                config.to_device(loaded_eval, target="numpy"),
                config.to_device(orig_eval, target="numpy"),
                rtol=1e-3,
                atol=1e-3,
            )

        # ---- 6. Explicit Target Device Migration Tests ----

        def test_explicit_device_migration_on_load(self):
            """Verify Model.load(..., device=target) places parameters onto requested backend."""
            model = Target_Class()
            model.add(Dense(self.NUM_FEATURES, 4))
            model.finalize((self.NUM_FEATURES,))
            model.save(self.save_path)

            loaded_model = Target_Class.load(self.save_path, device="numpy")
            self.assertEqual(loaded_model.device, "numpy")
            self.assertIsInstance(loaded_model.layers[0].weights, np.ndarray)

    TestModelLoad.__name__ = class_name
    TestModelLoad.__qualname__ = class_name
    return TestModelLoad


for backend in backends_to_test:
    class_name = f"Test_{TARGET_CLASS.__name__}_Load_{backend.upper()}"
    globals()[class_name] = make_suite(
        backend_name=backend, Target_Class=TARGET_CLASS
    )