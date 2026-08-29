import io
import contextlib
import unittest
import numpy as np

import aether.config as config
from aether.model import Model
from aether.layers import Dense, ReLU
from aether.losses import CategoricalCrossEntropy
from aether.optimizers import Adam, AdamW
from aether.metrics import CategoricalAccuracy

from tests.base_case import register_test_suites
from tests.model.base import ModelBaseTestCase


class TestModelTrainBase(ModelBaseTestCase):
    __test__ = False

    def setUp(self):
        super().setUp()
        self.model = Model()
        self.model.to(self.backend_name)
        self.model.add(Dense(n_inputs= 4, n_neurons=8))
        self.model.add(ReLU())
        self.model.add(Dense(n_inputs = 8, n_neurons=self.NUM_CLASSES))

        self.loss = CategoricalCrossEntropy()
        self.optimizer = Adam(lr=0.01)
        self.accuracy = CategoricalAccuracy()

        self.model.configure(
            loss=self.loss,
            optimizer=self.optimizer,
            accuracy=self.accuracy
        )

    def test_train_unfinalized_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            self.model.train(self.X, self.y, verbose=False)

    def test_train_without_loss_raises_runtime_error(self):
        model = Model()
        model.to(self.backend_name)
        model.add(Dense(n_inputs=4, n_neurons=self.NUM_CLASSES))
        model.configure(optimizer=self.optimizer, accuracy=self.accuracy)
        model.finalize(input_shape=(self.NUM_FEATURES,))

        with self.assertRaises(RuntimeError):
            model.train(self.X, self.y, verbose=False)

    def test_train_updates_layer_parameters(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        trainable_layer = self.model.trainable_layers[0]

        initial_weights = self.xp.copy(trainable_layer.weights)
        self.model.train(self.X, self.y, epochs=1, batch_size=None, verbose=False)

        self.assertFalse(
            self.xp.allclose(initial_weights, trainable_layer.weights),
            "Expected weights to update after a training step."
        )

    def test_train_mini_batch_multiple_epochs(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        trainable_layer = self.model.trainable_layers[0]
        initial_weights = self.xp.copy(trainable_layer.weights)

        self.model.train(
            self.X,
            self.y,
            epochs=3,
            batch_size=8,
            shuffle=True,
            verbose=False
        )

        self.assertFalse(self.xp.allclose(initial_weights, trainable_layer.weights))

    def test_train_without_shuffle(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        trainable_layer = self.model.trainable_layers[0]
        initial_weights = self.xp.copy(trainable_layer.weights)

        self.model.train(
            self.X,
            self.y,
            epochs=2,
            batch_size=16,
            shuffle=False,
            verbose=False
        )

        self.assertFalse(self.xp.allclose(initial_weights, trainable_layer.weights))

    def test_train_with_regularization(self):
        model = Model()
        model.to(self.backend_name)
        model.add(Dense(n_inputs=4, n_neurons=self.NUM_CLASSES, l2=(0.01)))
        model.configure(
            loss=CategoricalCrossEntropy(),
            optimizer=AdamW(lr=0.01),
            accuracy=CategoricalAccuracy()
        )
        model.finalize(input_shape=(self.NUM_FEATURES,))

        model.train(self.X, self.y, epochs=1, batch_size=16, verbose=False)

    def test_train_with_validation_data(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        self.model.train(
            self.X,
            self.y,
            epochs=1,
            batch_size=16,
            validation_data=(self.X_val, self.y_val),
            verbose=False
        )

    def test_train_device_mismatch_raises_type_error(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        if self.backend_name == "cupy":
            X_host = np.random.randn(self.NUM_SAMPLES, self.NUM_FEATURES).astype("float32")
            y_host = np.random.randint(0, self.NUM_CLASSES, size=(self.NUM_SAMPLES,)).astype("int32")
            with self.assertRaises(TypeError):
                self.model.train(X_host, y_host, verbose=False)
        elif config.HAS_CUPY:
            import cupy as cp
            X_gpu = cp.random.randn(self.NUM_SAMPLES, self.NUM_FEATURES).astype("float32")
            y_gpu = cp.random.randint(0, self.NUM_CLASSES, size=(self.NUM_SAMPLES,)).astype("int32")
            with self.assertRaises(TypeError):
                self.model.train(X_gpu, y_gpu, verbose=False)

    def test_train_verbose_output_suppression(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.model.train(self.X, self.y, epochs=1, verbose=False)
        self.assertEqual(buffer.getvalue(), "")


register_test_suites(globals(), TestModelTrainBase)