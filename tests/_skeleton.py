# Template for writing a test file for a layer
import unittest
import numpy as np

# 1. Base Import from CPU backend
from CNN.models.CNN_classes import Pooling  # Example layer class

# Initialize profiles list with the mandatory NumPy suite
test_profiles = [
    (np, Pooling)
]

# 2. Safe GPU check & loading loop
try:
    import cupy as cp
    # Assuming your GPU class maps cleanly or is imported from your cupy file
    from CNN.models.CNN_classes_cupy import Pooling as PoolingCuPy

    test_profiles.append((cp, PoolingCuPy))
except (ImportError, Exception):
    pass  # Gracefully fall back onto the NumPy suite if CUDA/ROCm isn't present


# 3. Dynamic Factory Class Generation
def make_suite(xp, Layer_Class):
    
    class TestLayer(unittest.TestCase):
        def setUp(self):
            # Attach the engine module and structural class to 'self'
            # Now every test function below can adapt dynamically!
            self.xp = xp
            self.Layer = Layer_Class

        def test_forward_shape(self):
            # Example write up using your universal engine syntax:
            x = self.xp.ones((2, 4, 4, 3), dtype=self.xp.float32)
            layer = self.Layer(filter_size=(2, 2), strides=(2, 2), padding="valid")
            output = layer.forward(x, training=True)
            
            self.assertEqual(output.shape, (2, 2, 2, 3))

        def test_backward_gradient(self):
            # Your complex backward pass assertions go here...
            pass

    return TestLayer


# 4. Global Unpacking Loop (Fixed bugs here)
for backend_xp, layer_cls in test_profiles:
    # Build the specific dynamic class instance
    layer_suite = make_suite(backend_xp, layer_cls)
    
    # Grab the string representation (__name__ converts np to 'numpy' and cp to 'cupy')
    backend_name = backend_xp.__name__

    # Bind the generated class to the global workspace module context
    # This generates "TestLayer_numpy" and "TestLayer_cupy"
    globals()[f"TestLayer_{backend_name}"] = layer_suite