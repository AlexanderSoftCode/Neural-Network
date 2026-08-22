"""
Simple null objects that are used by the Model class.
Avoids making us write needless branch conditionals.
"""
class NullAccuracy:
    def calculate(self, *args, **kwargs):
        return 0.0

    def calculate_accumulated(self):
        return 0.0

    def new_pass(self):
        pass


class NullOptimizer:
    def step(self):
        pass

    @property
    def current_learning_rate(self):
        return None