from aether._utils._lazy import install_lazy_attrs as install_lazy_attrs
from aether._utils.null_objects import (
    NullAccuracy as NullAccuracy,
    NullOptimizer as NullOptimizer,
)

__all__ = [
    "install_lazy_attrs",
    "NullAccuracy",
    "NullOptimizer",
]