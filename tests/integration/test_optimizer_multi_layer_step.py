from tests.base_case import register_test_suites
from tests.integration.model_base_suite import ModelIntegrationBaseCase


class TestOptimizerMultiLayerStep(ModelIntegrationBaseCase):

    N_SAMPLES = 16
    BATCH_SIZE = 8

    def setUp(self):
        super().setUp()
        self.X, self.y = self.make_synthetic_image_data(n_samples=self.N_SAMPLES)

    def test_all_cnn_trainable_layer_weights_update_after_one_epoch(self):
        model = self.build_cnn_model()
        before = [layer.weights.copy() for layer in model.trainable_layers]

        model.train(
            X=self.X,
            y=self.y,
            epochs=1,
            batch_size=self.BATCH_SIZE,
            verbose=False,
            print_every=10**9,
        )

        for prev, layer in zip(before, model.trainable_layers):
            with self.subTest(layer=type(layer).__name__):
                self.assertFalse(
                    bool(self.xp.allclose(prev, layer.weights)),
                    msg=f"{type(layer).__name__} weights did not change after training.",
                )

    def test_all_mlp_trainable_layer_weights_update_after_one_epoch(self):
        model = self.build_mlp_model()
        before = [layer.weights.copy() for layer in model.trainable_layers]

        model.train(
            X=self.X,
            y=self.y,
            epochs=1,
            batch_size=self.BATCH_SIZE,
            verbose=False,
            print_every=10**9,
        )

        for prev, layer in zip(before, model.trainable_layers):
            with self.subTest(layer=type(layer).__name__):
                self.assertFalse(bool(self.xp.allclose(prev, layer.weights)))

    def test_biases_also_update(self):
        model = self.build_mlp_model()
        before = [layer.biases.copy() for layer in model.trainable_layers]

        model.train(
            X=self.X,
            y=self.y,
            epochs=1,
            batch_size=self.BATCH_SIZE,
            verbose=False,
            print_every=10**9,
        )

        for prev, layer in zip(before, model.trainable_layers):
            with self.subTest(layer=type(layer).__name__):
                self.assertFalse(bool(self.xp.allclose(prev, layer.biases)))

    def test_weight_updates_stay_finite(self):
        """Catches divergence/NaN blowups that a naive weight-changed check
        would falsely accept (a NaN'd-out weight also 'changed')."""
        model = self.build_cnn_model()
        model.train(
            X=self.X,
            y=self.y,
            epochs=1,
            batch_size=self.BATCH_SIZE,
            verbose=False,
            print_every=10**9,
        )

        for layer in model.trainable_layers:
            with self.subTest(layer=type(layer).__name__):
                self.assertTrue(
                    bool(self.xp.all(self.xp.isfinite(layer.weights)))
                )
                self.assertTrue(
                    bool(self.xp.all(self.xp.isfinite(layer.biases)))
                )

    def test_multi_epoch_training_does_not_reset_weights(self):
        """Weights after 2 epochs should differ from weights after 1 epoch
        -- catches an optimizer/model wiring bug where state is silently
        re-initialized every epoch instead of carried forward."""
        model = self.build_mlp_model()

        model.train(
            X=self.X,
            y=self.y,
            epochs=1,
            batch_size=self.BATCH_SIZE,
            verbose=False,
            print_every=10**9,
        )
        after_epoch_1 = [
            layer.weights.copy() for layer in model.trainable_layers
        ]

        model.train(
            X=self.X,
            y=self.y,
            epochs=1,
            batch_size=self.BATCH_SIZE,
            verbose=False,
            print_every=10**9,
        )
        after_epoch_2 = [layer.weights for layer in model.trainable_layers]

        for w1, w2, layer in zip(
            after_epoch_1, after_epoch_2, model.trainable_layers
        ):
            with self.subTest(layer=type(layer).__name__):
                self.assertFalse(bool(self.xp.allclose(w1, w2)))


register_test_suites(globals(), TestOptimizerMultiLayerStep)