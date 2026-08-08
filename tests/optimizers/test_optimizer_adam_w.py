from aether.optimizers.adam import Optimizer_AdamW
from tests.optimizers.adam_base_suite import make_suite, backends_to_test

TARGET_LAYER = Optimizer_AdamW

for backend in backends_to_test:
    # 2. Generate the base class for AdamW
    suite_cls = make_suite(backend_name=backend, Optimizer_Class=Optimizer_AdamW)
    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"

    def test_weight_decay_decoupled_update(self):
        """Verify decoupled weight decay behavior."""
        self.layer.dweights = self.xp.zeros_like(self.layer.weights)
        self.layer.dbiases = self.xp.zeros_like(self.layer.biases)

        self.optimizer.weight_decay = 0.1
        lr = self.optimizer.current_learning_rate
        weights_before = self.layer.weights.copy()

        self.optimizer.update_parameters(self.layer)

        expected_weights = weights_before - (lr * 0.1 * weights_before)
        self.xp.testing.assert_array_almost_equal(
            self.layer.weights, expected_weights, decimal=5
        )

    def test_no_weight_decay_flag_bypasses_decay(self):
        """Verify layer-level no_weight_decay flag."""
        self.layer.no_weight_decay = True
        self.layer.dweights = self.xp.zeros_like(self.layer.weights)
        self.optimizer.weight_decay = 0.1
        weights_before = self.layer.weights.copy()

        self.optimizer.update_parameters(self.layer)
        self.xp.testing.assert_array_equal(self.layer.weights, weights_before)

    suite_cls.test_weight_decay_decoupled_update = test_weight_decay_decoupled_update
    suite_cls.test_no_weight_decay_flag_bypasses_decay = test_no_weight_decay_flag_bypasses_decay

    suite_cls.__name__ = class_name
    suite_cls.__qualname__ = class_name
    suite_cls.__module__ = __name__

    globals()[class_name] = suite_cls


del suite_cls