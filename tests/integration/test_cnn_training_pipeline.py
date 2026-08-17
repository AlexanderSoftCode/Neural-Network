import aether.config as config
from tests.integration.model_base_suite import ModelIntegrationBaseCase, backends_to_test

def make_suite(backend_name):
    class_name = f"Test_CnnTrainingPipeline_{backend_name.upper()}"

    class TestCnnTrainingPipeline(ModelIntegrationBaseCase):
        N_SAMPLES = 16
        BATCH_SIZE = 4

        def setUp(self):
            self.backend_name = backend_name
            config.set_backend(backend_name=backend_name)
            self.xp = config.xp
            self.X, self.y = self.make_synthetic_image_data(n_samples=self.N_SAMPLES)

        def test_train_runs_and_produces_finite_loss(self):
            model = self.build_cnn_model(device=backend_name)
            model.train(X=self.X, y=self.y, epochs=1, batch_size=self.BATCH_SIZE,
                        verbose=False, print_every=10 ** 9)
            val_loss, val_acc = model.evaluate(self.X, self.y, batch_size=self.BATCH_SIZE, verbose=False)

            self.assertTrue(bool(self.xp.isfinite(val_loss)))
            self.assertGreaterEqual(val_acc, 0.0)
            self.assertLessEqual(val_acc, 1.0)

        def test_gradients_reach_conv_and_dense_layers(self):
            """White-box check: verifies gradients propagate through all trainable 
            Conv and Dense layers using raw unnormalized logits (matching Model.train)."""
            model = self.build_cnn_model(device=backend_name)

            # 1. Forward pass returns raw logits
            out = model.forward(self.X, training=True)
            model.loss.calculate(out, self.y)

            # 2. Pass 'out' (raw logits) directly into fused loss backward
            model.loss.backward(out, self.y)
            model.backward(model.loss.dinputs)

            # 3. Verify gradients on all trainable layers (Conv, Dense, etc.)
            for layer in model.trainable_layers:
                with self.subTest(layer=type(layer).__name__):
                    self.assertTrue(hasattr(layer, "dweights"))
                    self.assertEqual(layer.dweights.shape, layer.weights.shape)
                    self.assertTrue(bool(self.xp.all(self.xp.isfinite(layer.dweights))))

                    if hasattr(layer, "biases") and layer.biases is not None:
                        self.assertTrue(hasattr(layer, "dbiases"))
                        self.assertEqual(layer.dbiases.shape, layer.biases.shape)
                        self.assertTrue(bool(self.xp.all(self.xp.isfinite(layer.dbiases))))

        def test_predict_batched_matches_unbatched(self):
            model = self.build_cnn_model(device=backend_name)
            full = model.predict(self.X, batch_size=None)
            batched = model.predict(self.X, batch_size=self.BATCH_SIZE)
            # Looser tolerance than the MLP equivalent: on GPU, Conv's
            # matrix-core forward runs fp16 regardless of batch slicing, so
            # small numeric drift between the two batchings is expected.
            self.xp.testing.assert_allclose(full, batched, rtol=1e-2, atol=1e-2)

        def test_fp16_precision_forward_produces_finite_output_on_gpu(self):
            """Conv's GPU matrix-core forward always runs fp16 internally
            regardless of the model's precision policy (Conv has no
            _apply_precision override -- see Conv._forward_gpu casting inputs
            unconditionally). This test only confirms the composition survives
            set_precision('float16') end-to-end through Dense + Conv together,
            not that Conv's behavior changes because of it."""
            if self.backend_name != 'cupy':
                self.skipTest("fp16 compute path is only meaningfully exercised on the CuPy backend.")

            model = self.build_cnn_model(device='cupy', precision='float16')
            out = model.forward(self.X, training=False)
            self.assertTrue(bool(self.xp.all(self.xp.isfinite(out))))

        def test_spatial_dropout_changes_output_between_train_and_eval(self):
            """Sanity check that training=True/False actually routes through
            SpatialDropout differently, since that only shows up once it's
            embedded in a full Model.forward() call, not in isolation."""
            model = self.build_cnn_model(device=backend_name)
            train_out_1 = model.forward(self.X, training=True)
            train_out_2 = model.forward(self.X, training=True)
            eval_out_1 = model.forward(self.X, training=False)
            eval_out_2 = model.forward(self.X, training=False)

            self.assertFalse(bool(self.xp.allclose(train_out_1, train_out_2)))
            self.xp.testing.assert_allclose(eval_out_1, eval_out_2, rtol=1e-5, atol=1e-6)

        def test_loss_decreases_over_training_steps(self):
            """Composition-level check driven through the public Model.train()
            API on a small fixed batch. If this fails, check how run_step()
            feeds values into loss.backward() for the fused
            SoftmaxCategoricalCrossEntropy path -- see
            test_model_finalize_wiring.py notes."""
            model = self.build_cnn_model(device=backend_name)

            model.train(X=self.X, y=self.y, epochs=1, batch_size=self.BATCH_SIZE,
                        verbose=False, print_every=10 ** 9)
            first_loss, _ = model.evaluate(self.X, self.y, batch_size=self.BATCH_SIZE, verbose=False)

            model.train(X=self.X, y=self.y, epochs=15, batch_size=self.BATCH_SIZE,
                        verbose=False, print_every=10 ** 9)
            later_loss, _ = model.evaluate(self.X, self.y, batch_size=self.BATCH_SIZE, verbose=False)

            self.assertLess(later_loss, first_loss)

    TestCnnTrainingPipeline.__name__ = class_name
    TestCnnTrainingPipeline.__qualname__ = class_name
    return TestCnnTrainingPipeline


for backend in backends_to_test:
    globals()[f"Test_CnnTrainingPipeline_{backend.upper()}"] = make_suite(backend)