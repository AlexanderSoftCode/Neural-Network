from aether.optimizers.adam import AdamW
from tests.optimizers.adam_base_suite import make_suite, backends_to_test

TARGET_LAYER = AdamW

for backend in backends_to_test:
    # 2. Generate the base class for AdamW
    suite_cls = make_suite(backend_name=backend, Optimizer_Class=AdamW)
    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"

    def test_weight_decay_decoupled_update(self):
        """Verify decoupled weight decay behavior when gradients are zero."""
        self.layer.dweights = self.xp.zeros_like(self.layer.weights)
        self.layer.dbiases = self.xp.zeros_like(self.layer.biases)

        self.optimizer.weight_decay = 0.1
        learning_rate = self.optimizer.current_learning_rate
        weights_before = self.layer.weights.copy()
        biases_before = self.layer.biases.copy()

        self.optimizer.step()

        # With zero gradients, parameters should change strictly due to decoupled decay
        expected_weights = weights_before - (learning_rate * 0.1 * weights_before)
        self.xp.testing.assert_array_almost_equal(
            self.layer.weights, expected_weights, decimal=5
        )
        
        # Biases must never be subject to weight decay
        self.xp.testing.assert_array_equal(self.layer.biases, biases_before)

    def test_no_weight_decay_flag_bypasses_decay(self):
        """Verify layer-level no_weight_decay flag bypasses decoupled weight decay."""
        self.layer.no_weight_decay = True
        self.layer.dweights = self.xp.zeros_like(self.layer.weights)
        self.layer.dbiases = self.xp.zeros_like(self.layer.biases)
        self.optimizer.weight_decay = 0.1
        weights_before = self.layer.weights.copy()

        self.optimizer.step()

        self.xp.testing.assert_array_equal(self.layer.weights, weights_before)

    suite_cls.test_weight_decay_decoupled_update = test_weight_decay_decoupled_update
    suite_cls.test_no_weight_decay_flag_bypasses_decay = test_no_weight_decay_flag_bypasses_decay

    suite_cls.__name__ = class_name
    suite_cls.__qualname__ = class_name
    suite_cls.__module__ = __name__

    globals()[class_name] = suite_cls


del suite_cls