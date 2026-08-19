# Static declaration for the IDEs
from aether.layers.linear import Dense as Dense, Flatten as Flatten
from aether.layers.conv import Conv as Conv
from aether.layers.activations import ReLU as ReLU, LeakyReLU as LeakyReLU, SoftMax as SoftMax
from aether.layers.pooling import MaxPool2d as MaxPool2d, AvgPool2d as AvgPool2d, GlobalAvgPool as GlobalAvgPool
from aether.layers.dropout import Dropout as Dropout, SpatialDropout as SpatialDropout
from aether.layers.normalization import BatchNorm as BatchNorm
__all__ = [
    "Dense",
    "Flatten",
    "Conv",
    "ReLU",
    "LeakyReLU",
    "Softmax",
    "MaxPool2d",
    "AvgPool2d",
    "GlobalAvgPool",
    "Dropout",
    "SpatialDropout",
    "BatchNorm",
]