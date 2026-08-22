from aether._utils._lazy import install_lazy_attrs
from aether.layers import _LAYER_MODULES
from aether.losses import _LOSS_MODULES
from aether.metrics import _ACC_MODULES
from aether.optimizers import _OPTIMIZER_MODULES
from aether.preprocessing import _PREPROCESSING_MODULES

_MODEL_MODULES = {
    "Model": "aether.model",
}
_TOP_LEVEL_MODULES = {
    **_MODEL_MODULES,
    **_LAYER_MODULES,
    **_LOSS_MODULES,
    **_ACC_MODULES,
    **_OPTIMIZER_MODULES,
    **_PREPROCESSING_MODULES
}
 
install_lazy_attrs(globals(), _TOP_LEVEL_MODULES) 

# Clean up private setup variables so ae.<tab> remains clean
del (
    install_lazy_attrs,
    _LAYER_MODULES, 
    _LOSS_MODULES, 
    _ACC_MODULES,
    _OPTIMIZER_MODULES, 
    _MODEL_MODULES, 
    _PREPROCESSING_MODULES,
    _TOP_LEVEL_MODULES,
)