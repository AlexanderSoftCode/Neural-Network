from aether._lazy import install_lazy_attrs

_OPTIMIZER_MODULES = {
    "Adam":  "aether.optimizers.adam",
    "AdamW": "aether.optimizers.adam"
}

install_lazy_attrs(globals(), _OPTIMIZER_MODULES)