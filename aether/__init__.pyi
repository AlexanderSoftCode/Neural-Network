# aether/__init__.pyi

# Static-typing companion to aether/__init__.py.
# Never executed -- exists purely so each public class
# gets real autocomplete/type info despite the runtime 
# __getattr__ lazy loading.

from aether.model import Model as Model

from aether.layers.linear import Dense as Dense, Flatten as Flatten
from aether.layers.conv import Conv as Conv
from aether.layers.activations import ReLU as ReLU, LeakyReLU as LeakyReLU, SoftMax as SoftMax
from aether.layers.pooling import MaxPool2d as MaxPool2d, AvgPool2d as AvgPool2d, GlobalAvgPool as GlobalAvgPool
from aether.layers.dropout import Dropout as Dropout, SpatialDropout as SpatialDropout

from aether.losses.categorical_crossentropy import (
    Loss as Loss,
    CategoricalCrossEntropy as CategoricalCrossEntropy,
    SoftmaxCategoricalCrossEntropy as SoftmaxCategoricalCrossEntropy
)

# Direct module imports resolve stub lookups instantly:
from aether.metrics.accuracy import (
    CategoricalAccuracy as CategoricalAccuracy,
    RegressionAccuracy as RegressionAccuracy,
)

from aether.optimizers.adam import Adam as Adam, AdamW as AdamW

from aether.preprocessing.transforms import (
    to_tensor as to_tensor, 
    ToTensor as ToTensor, 
    StandardScaler as StandardScaler,
    Rescale as Rescale,
    Compose as Compose
)
__all__ = [
    # model
    "Model",
    # layers
    "Dense",
    "Flatten",
    "Conv",
    "ReLU",
    "LeakyReLU",
    "SoftMax",
    "MaxPool2d",
    "AvgPool2d",
    "GlobalAvgPool",
    "Dropout",
    "SpatialDropout",
    # losses
    "Loss",
    "CategoricalCrossEntropy",
    "SoftmaxCategoricalCrossEntropy",
    # metrics
    "CategoricalAccuracy",
    "RegressionAccuracy",
    # optimizers
    "Adam",
    "AdamW",
    # preprocessing
    "to_tensor",
    "ToTensor",
    "StandardScaler",
    "Rescale",
    "Compose"
]