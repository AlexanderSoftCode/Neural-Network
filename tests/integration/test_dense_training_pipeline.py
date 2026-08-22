from tests.base_case import register_test_suites
from tests.integration.model_base_suite import ModelIntegrationBaseCase


class TestDenseTrainingPipeline(ModelIntegrationBaseCase):
    N_SAMPLES = 32
    BATCH_SIZE = 8

    def setUp(self):
        super().setUp()
        self.X, self.y = self.make_synthetic_image_data(n_samples=self.N_SAMPLES)

    def test_train_runs_without_error_and_produces_finite_loss(self):
        model = self.build_mlp_model()
        model.train(
            X=self.X,
            y=self.y,
            epochs=1,
            batch_size=self.BATCH_SIZE,
            verbose=False,
            print_every=10**9,
        )
        val_loss, val_acc = model.evaluate(
            self.X, self.y, batch_size=self.BATCH_SIZE, verbose=False
        )

        self.assertTrue(bool(self.xp.isfinite(val_loss)))
        self.assertGreaterEqual(val_acc, 0.0)
        self.assertLessEqual(val_acc, 1.0)

    def test_gradients_reach_every_trainable_layer(self):
        """White-box check: verifies gradients propagate through all trainable layers."""
        model = self.build_mlp_model()

        out = model.forward(self.X, training=True)
        model.loss.calculate(out, self.y)

        # Pass 'out' directly (matching Model.train)
        model.loss.backward(out, self.y)
        model.backward(model.loss.dinputs)

        for layer in model.trainable_layers:
            with self.subTest(layer=type(layer).__name__):
                self.assertTrue(hasattr(layer, "dweights"))
                self.assertTrue(hasattr(layer, "dbiases"))
                self.assertEqual(layer.dweights.shape, layer.weights.shape)
                self.assertEqual(layer.dbiases.shape, layer.biases.shape)
                self.assertTrue(bool(self.xp.all(self.xp.isfinite(layer.dweights))))
                self.assertTrue(bool(self.xp.all(self.xp.isfinite(layer.dbiases))))

    def test_predict_batched_matches_unbatched(self):
        model = self.build_mlp_model()
        full = model.predict(self.X, batch_size=None)
        batched = model.predict(self.X, batch_size=self.BATCH_SIZE)
        self.xp.testing.assert_allclose(full, batched, rtol=1e-4, atol=1e-5)

    def test_predict_returns_probabilities_by_default(self):
        """predict() should route through the fused loss's SoftMax activation
        unless return_logits=True, so rows should sum to ~1."""
        model = self.build_mlp_model()
        probs = model.predict(self.X, batch_size=self.BATCH_SIZE)
        row_sums = self.xp.sum(probs, axis=1)
        self.xp.testing.assert_allclose(
            row_sums, self.xp.ones_like(row_sums), rtol=1e-3, atol=1e-3
        )

    def test_predict_return_logits_skips_softmax(self):
        model = self.build_mlp_model()
        logits = model.predict(self.X, batch_size=self.BATCH_SIZE, return_logits=True)
        row_sums = self.xp.sum(logits, axis=1)

        self.assertFalse(bool(self.xp.allclose(row_sums, self.xp.ones_like(row_sums))))

    def test_unfinalized_model_raises_runtime_error_on_execution(self):
        """train()/evaluate()/predict() should raise a RuntimeError
        if the model has not been explicitly finalized with an input_shape."""
        import aether as ae

        model = ae.Model()
        model.add(ae.Flatten())
        model.add(ae.Dense(32 * 32 * 3, 16))
        model.add(ae.ReLU())
        model.add(ae.Dense(16, self.NUM_CLASSES))
        model.configure(
            loss=ae.SoftmaxCategoricalCrossEntropy(),
            optimizer=ae.Adam(),
            accuracy=ae.CategoricalAccuracy(),
        )
        model.to(self.backend_name)

        self.assertFalse(model.is_finalized)
        with self.assertRaises(RuntimeError):
            model.train(
                X=self.X,
                y=self.y,
                epochs=1,
                batch_size=self.BATCH_SIZE,
                verbose=False,
            )

        with self.assertRaises(RuntimeError):
            model.evaluate(
                self.X, self.y, batch_size=self.BATCH_SIZE, verbose=False
            )

        with self.assertRaises(RuntimeError):
            model.predict(self.X, batch_size=self.BATCH_SIZE)

    def test_loss_decreases_over_training_steps(self):
        """Composition-level check: forward -> loss -> backward -> optimizer,
        driven entirely through the public Model.train() API, should reduce
        loss on a small fixed batch over several epochs."""
        model = self.build_mlp_model()

        model.train(
            X=self.X,
            y=self.y,
            epochs=1,
            batch_size=self.BATCH_SIZE,
            verbose=False,
            print_every=10**9,
        )
        first_loss, _ = model.evaluate(
            self.X, self.y, batch_size=self.BATCH_SIZE, verbose=False
        )

        model.train(
            X=self.X,
            y=self.y,
            epochs=25,
            batch_size=self.BATCH_SIZE,
            verbose=False,
            print_every=10**9,
        )
        later_loss, _ = model.evaluate(
            self.X, self.y, batch_size=self.BATCH_SIZE, verbose=False
        )

        self.assertLess(later_loss, first_loss)


register_test_suites(globals(), TestDenseTrainingPipeline)