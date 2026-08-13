from aether._lazy import install_lazy_attrs

_ACC_MODULES = {
    "CategoricalAccuracy":     "aether.metrics.accuracy",
    "RegressionAccuracy":      "aether.metrics.accuracy"
}

install_lazy_attrs(globals(), _ACC_MODULES)