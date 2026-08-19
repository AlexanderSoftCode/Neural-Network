"""
Shared fixture-building helpers for Model-level integration tests.
"""
import math
import aether as ae
import aether.config as config
from tests.base_case import AetherBaseTestCase

backends_to_test = ['numpy']
try:
    import cupy as cp
    backends_to_test.append('cupy')
except (ImportError, Exception):
    pass


class ModelIntegrationBaseCase(AetherBaseTestCase):
    __test__ = False

    NUM_CLASSES = 10
    SEED = 42

    # ---- synthetic data -------------------------------------------------

    def make_synthetic_image_data(self, n_samples=16, height=32, width=32, channels=3):
        # Reads the active backend set by config.set_backend(), which may be overwritten during a unit-test
        xp = config.xp
        xp.random.seed(self.SEED)
        X = xp.random.randn(n_samples, height, width, channels).astype(xp.float32)
        y = xp.random.randint(0, self.NUM_CLASSES, size=(n_samples,)).astype(xp.int64)
        return X, y
    
    # ---- verified architectures ------------------------------------------

    def build_cnn_model(self, *, precision=None, device=None, input_dim=(32, 32, 3)):
        """
        Verified CNN architecture:
        Conv -> MaxPool2d -> ReLU -> Conv -> AvgPool2d -> LeakyReLU -> SpatialDropout ->
        GlobalAvgPool -> Dense, trained with the fused
        SoftmaxCategoricalCrossEntropy loss (no explicit SoftMax layer).
        """
        model = ae.Model()
        model.add(ae.Conv(3, 32, (3, 3), (1, 1), padding="same"))
        model.add(ae.MaxPool2d((2, 2), (2, 2), padding="valid"))
        model.add(ae.ReLU())
        model.add(ae.Conv(32, 64, (3, 3), (1, 1), padding="same"))
        model.add(ae.AvgPool2d((2,2), (2, 2), padding="valid"))
        model.add(ae.LeakyReLU(alpha=0.01))
        model.add(ae.SpatialDropout(rate=0.05, seed=self.SEED))
        model.add(ae.GlobalAvgPool())
        model.add(ae.Dense(64, self.NUM_CLASSES))
        model.configure(
            loss=ae.SoftmaxCategoricalCrossEntropy(label_smoothing=0.01),
            optimizer=ae.Adam(learning_rate=0.001, decay=5e-5),
            accuracy=ae.CategoricalAccuracy(),
        )
        return self._place_and_finalize(model, precision=precision, device=device, input_dim=input_dim)

    def build_mlp_model(self, *, precision=None, device=None, input_dim=(32, 32, 3)):
        """
        Verified MLP architecture:
        Flatten -> Dense -> ReLU -> Dense, trained with the fused
        SoftmaxCategoricalCrossEntropy loss.
        """
        model = ae.Model()
        model.add(ae.Flatten())
        model.add(ae.Dense(math.prod(input_dim), 128))
        model.add(ae.ReLU())
        model.add(ae.Dense(128, self.NUM_CLASSES))
        model.configure(
            loss=ae.SoftmaxCategoricalCrossEntropy(label_smoothing=0.01),
            optimizer=ae.Adam(learning_rate=0.001, decay=5e-5),
            accuracy=ae.CategoricalAccuracy(),
        )
        return self._place_and_finalize(model, precision=precision, device=device, input_dim=input_dim)

    @staticmethod
    def _place_and_finalize(model, *, precision, device, input_dim):

        if device is not None:
            model.to(device)

        if precision is not None:
            model.set_precision(compute_dtype=precision)
        model.finalize(input_shape=input_dim)
        return model