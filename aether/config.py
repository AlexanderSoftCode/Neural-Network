import numpy as np
import warnings

# default backend incase user does not have CuPy package
xp = np
as_strided = np.lib.stride_tricks.as_strided
HAS_CUPY = False

try:
    import cupy as cp
    HAS_CUPY = True
    get_array_module = cp.get_array_module

except (ImportError, ModuleNotFoundError):
    HAS_CUPY = False
    cp = None

    def get_array_module(*args, **kwargs):
        return np

xp = cp if HAS_CUPY else np
DEFAULT_DTYPE = np.float32

def enforce_cupy_env():
    """Loudly explodes only if a layer or user explicitly demands GPU execution."""
    if not HAS_CUPY:
        raise RuntimeError(
            "Failed to leverage backend logic for 'cupy'. "
            "Ensure that CuPy is installed and configured to your GPU environment.\n"
            "To install CuPy for AMD ROCm/NVIDIA CUDA, run: pip install cupy"
        )

def to_device(array, target='cupy'):
    """Explicity handles structural device migration outside execution passes."""

    if target.lower() == 'cupy':
        enforce_cupy_env()  # Explosive check triggers
        return cp.asarray(array)

    elif target.lower() == 'numpy':
        if HAS_CUPY and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)

def get_stride_utility(xp_module):
    """Unified helper for nested stride pathways."""
    return xp_module.lib.stride_tricks.as_strided

def set_backend(backend_name):
    """Dynamically switches the global array namespace and layout views at runtime."""
    global xp, as_strided
    if backend_name.lower() == 'cupy':
        enforce_cupy_env()
        xp = cp
        as_strided = cp.lib.stride_tricks.as_strided
    elif backend_name.lower() == 'numpy':
        xp = np
        as_strided = np.lib.stride_tricks.as_strided
    else:
        raise ValueError(f"Unknown backend configuration context: {backend_name}")

def fuse_kernel(*fuse_args, **fuse_kwargs):
    """
    Decorator for cp.fuse'd elementwise kernels. Binds the name to a
    working fused callable when CuPy is available and custruction succeeds.
    """

    def decorator(func):
        if not HAS_CUPY:
            return None
        try:
            return cp.fuse(*fuse_args, **fuse_kwargs)(func)
        except Exception as e:
            warnings.warn(f"[aether] fuse_kernel: '{func.__name}' failed to compile \n {e} ")
            return None

def build_kernel(factory, name=None):
    """
    Used for kernel objects that aren't a decorated function ex.ElementwiseKernel,
    cp.ReductionKernel, cp.RawKernel. 'factory' is a zero-arg callable that performs
    the actual construction, deferred so it's never evaluated while cp is None.
    
    EX: _leaky_relu_ew = config.build_kernel(lambda: config.cp.ElementwiseKernel(
    ...), 
    )
    """
    if not HAS_CUPY:
        return None
    try:
        return factory()
    except Exception as e:
        warnings.warn(f"[aether] build_kernel: '{name or getattr(factory, '__name__', '?')}' failed to compile: {e}")
        return None
