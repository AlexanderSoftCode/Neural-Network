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

class _NullProgress:
    """
    No-op stand-in for ``TrainingProgress`` used when ``verbose == 0``.
    """

    __slots__ = ()

    def start_epoch(self, epoch: int) -> None:
        return

    def tick(self, step: int, force: bool = False) -> None:
        return

    def update_metrics(
        self,
        step: int,
        loss: float,
        acc: float,
        lr: float,
        reg_loss: float | None = None,
    ) -> None:
        return

    def commit_epoch(
        self,
        epoch: int,
        epoch_loss: float,
        epoch_acc: float,
        lr: float,
    ) -> None:
        return

    def commit_validation(self, val_loss: float, val_acc: float) -> None:
        return

    def close(self) -> None:
        return