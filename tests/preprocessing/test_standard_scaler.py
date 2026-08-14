import numpy as np

import aether.config as config
from tests.base_case import AetherBaseTestCase
from aether.preprocessing.transforms import StandardScaler

TARGET_CLASS = StandardScaler

backends_to_test = ['numpy']
try:
    import cupy as cp
    backends_to_test.append('cupy')
except (ImportError, Exception):
    pass


def make_suite(backend_name, Scaler_Class):
    class_name = f"Test_{Scaler_Class.__name__}_{backend_name.upper()}"

    class TestStandardScalerTransform(AetherBaseTestCase):

        def setUp(self):
            self.backend_name = backend_name
            config.set_backend(backend_name=self.backend_name)
            self.xp = config.xp

        def test_unfit_scaler_raises_error(self):
            """Calling __call__ without prior fit() or pre-computed stats must raise ValueError."""
            scaler = Scaler_Class()
            X = self.xp.array([1.0, 2.0, 3.0], dtype=self.xp.float32)

            with self.assertRaises(ValueError):
                scaler(X)

        def test_fit_calculates_correct_statistics_and_chains(self):
            """fit() should compute accurate mean/std and return self for method chaining."""
            scaler = Scaler_Class()
            X = self.xp.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=self.xp.float32)

            returned_instance = scaler.fit(X)

            self.assertIs(returned_instance, scaler)

            self.assertAlmostEqual(float(scaler.mean.item()), 30.0, places=4)
            self.assertAlmostEqual(float(scaler.std.item()), 14.1421356, places=4)

        def test_fit_integer_dtype_conversion(self):
            """Integer input arrays should default target dtype to float32 upon fitting."""
            scaler = Scaler_Class()
            X_int = self.xp.array([0, 128, 255], dtype=self.xp.uint8)

            scaler.fit(X_int)

            self.assertEqual(scaler.dtype, np.float32)

        def test_standardization_zero_mean_unit_variance(self):
            """Scaling data should yield approximately zero mean and unit variance."""
            scaler = Scaler_Class()

            X = self.xp.random.randn(1000, 16).astype(self.xp.float32) * 5.0 + 10.0

            scaler.fit(X)
            X_scaled = scaler(X)

            scaled_mean = float(self.xp.mean(X_scaled))
            scaled_std = float(self.xp.std(X_scaled))

            self.assertAlmostEqual(scaled_mean, 0.0, places=3)
            self.assertAlmostEqual(scaled_std, 1.0, places=3)

        def test_precomputed_mean_and_std_initialization(self):
            """Initializing with explicit mean/std allows scaling without running fit()."""
            scaler = Scaler_Class(mean=10.0, std=2.0)
            scaler.dtype = self.xp.float32

            X = self.xp.array([10.0, 12.0, 6.0], dtype=self.xp.float32)
            X_scaled = scaler(X)

            # (10 - 10) / 2 = 0;  (12 - 10) / 2 = 1;  (6 - 10) / 2 = -2
            expected = self.xp.array([0.0, 1.0, -2.0], dtype=self.xp.float32)
            
            self.assertTrue(self.xp.allclose(X_scaled, expected, atol=1e-5))

        def test_input_dtype_coercion_on_call(self):
            """Inputs with a dtype mismatched from fitted scaler.dtype should be safely casted."""
            scaler = Scaler_Class()
            X_train = self.xp.array([1.0, 2.0, 3.0], dtype=self.xp.float32)
            scaler.fit(X_train)

            # Pass mismatched float64 array during inference
            X_test_float64 = self.xp.array([1.0, 2.0, 3.0], dtype=self.xp.float64)
            X_scaled = scaler(X_test_float64)

            self.assertEqual(str(X_scaled.dtype), 'float32')

        def test_standard_scaler_default_axis_none(self):
            """Verify that default axis=None computes global scalar mean/std while keeping dimensions."""
            scaler = Scaler_Class(axis=None)
            X = self.xp.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.xp.float32)
            scaler.fit(X)

            # Global mean is (1 + 2 + 3 + 4) / 4 = 2.5
            self.assertEqual(scaler.mean.shape, (1, 1))
            self.assertEqual(scaler.std.shape, (1, 1))
            self.assertAlmostEqual(float(scaler.mean[0, 0]), 2.5, places=5)

            out = scaler(X)
            self.assertEqual(out.shape, X.shape)
            self.assertAlmostEqual(float(self.xp.mean(out)), 0.0, places=5)
            self.assertAlmostEqual(float(self.xp.std(out)), 1.0, places=5)

        def test_standard_scaler_tabular_axis_zero(self):
            """Verify feature-wise standardization for 2D tabular data using axis=0."""
            scaler = Scaler_Class(axis=0)
            # 2 samples, 3 features
            X = self.xp.array([
                [10.0, 100.0, 1.0],
                [20.0, 200.0, 3.0]
            ], dtype=self.xp.float32)
            scaler.fit(X)

            # Feature means: [15.0, 150.0, 2.0]
            self.assertEqual(scaler.mean.shape, (1, 3))
            self.assertEqual(scaler.std.shape, (1, 3))
            expected_means = self.xp.array([[15.0, 150.0, 2.0]], dtype=self.xp.float32)
            self.assertTrue(self.xp.allclose(scaler.mean, expected_means, atol=1e-5))

            out = scaler(X)
            # Each column should now have zero mean
            col_means = self.xp.mean(out, axis=0)
            self.assertTrue(self.xp.allclose(col_means, self.xp.zeros((3,), dtype=self.xp.float32), atol=1e-5))

        def test_standard_scaler_channel_wise_image_axis(self):
            """Verify channel-wise standardization for 4D image batches using axis=(0, 1, 2)."""
            scaler = Scaler_Class(axis=(0, 1, 2))
            # Batch of 4 images, 8x8, 3 channels (RGB)
            X = self.xp.random.uniform(0.0, 1.0, size=(4, 8, 8, 3)).astype(self.xp.float32)
            scaler.fit(X)

            # Shape should preserve rank with shape (1, 1, 1, 3)
            self.assertEqual(scaler.mean.shape, (1, 1, 1, 3))
            self.assertEqual(scaler.std.shape, (1, 1, 1, 3))

            out = scaler(X)
            self.assertEqual(out.shape, (4, 8, 8, 3))
            self.assertEqual(str(out.dtype), 'float32')

            # Channel means across (batch, height, width) should be zero
            channel_means = self.xp.mean(out, axis=(0, 1, 2))
            self.assertTrue(self.xp.allclose(channel_means, self.xp.zeros((3,), dtype=self.xp.float32), atol=1e-5))

        def test_input_dtype_coercion_on_call(self):
            """Inputs with a dtype mismatched from fitted scaler.dtype should be safely casted."""
            scaler = Scaler_Class()
            X_train = self.xp.array([1.0, 2.0, 3.0], dtype=self.xp.float32)
            scaler.fit(X_train)

            # Pass mismatched float64 array during inference
            X_test_float64 = self.xp.array([1.0, 2.0, 3.0], dtype=self.xp.float64)
            X_scaled = scaler(X_test_float64)

            self.assertEqual(str(X_scaled.dtype), 'float32')
    TestStandardScalerTransform.__name__ = class_name
    TestStandardScalerTransform.__qualname__ = class_name

    return TestStandardScalerTransform


# Global Unpacking Loop for Test Runner Discovery
for backend in backends_to_test:
    class_name = f"Test_{TARGET_CLASS.__name__}_{backend.upper()}"
    globals()[class_name] = make_suite(backend_name=backend, Scaler_Class=TARGET_CLASS)