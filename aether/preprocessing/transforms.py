import numpy as np
from typing import Sequence, Union, Tuple, Optional, Any
import aether.config as config
from ._utils import parse_inputs, resolve_dtypes, convert_single_tensor

ALLOWED_DTYPES = {'float16', 'bfloat16', 'float32', 'float64'}

def to_tensor(
    *args,
    target_device: str | None = None,
    dtype: Union[str, type, Sequence] | None = None,
    preserve_integers: bool = True
) -> Union[Any, Tuple[Any, ...], None]:
    """Converts array-like inputs into backend tensors (NumPy or CuPy).

    Accepts individual arrays or multiple positional arrays (e.g., training/testing sets,
    feature/label splits) and casts them based on provided specifications.

    Args:
        *args: One or more array-like objects to convert. May optionally include
            a trailing positional argument specifying target `dtype`(s).
        target_device: Hardware execution backend ('numpy' or 'cupy'). If None,
            defaults to the active device specified in `aether.config`.
        dtype: Desired floating-point data type. Can be passed as a single dtype
            (broadcasted across all arrays) or a sequence/tuple matching the number
            of passed arrays 1:1.
        preserve_integers: If True, bypasses `dtype` casting for integer-based arrays
            (e.g., target labels or indices) to prevent array indexing bugs.

    Returns:
        Converted backend tensor if a single input array was provided, a tuple of
        converted backend tensors if multiple inputs were provided, or None if input
        is empty.

    Raises:
        ValueError: If `dtype` tuple length does not match the number of input arrays,
            or if an unsupported `dtype` string/type is provided.

    Warns:
        UserWarning: If `float64` precision is specified, cautioning against hardware
            performance degradation.

    Examples:
        >>> import aether as ae
        >>> X_tr, y_tr = ae.to_tensor(X_tr, y_tr, dtype='float32')
        >>> X_tr, X_ts, y_tr, y_ts = ae.to_tensor(X_tr, X_ts, y_tr, y_ts, dtype=('float32', 'float32', None, None))
    """

    arrays, raw_dtype = parse_inputs(args, dtype)
    if not arrays:
        return None

    if target_device is None:
        target_device = 'cupy' if (config.HAS_CUPY and config.xp == config.cp) else 'numpy'

    target_dtypes = resolve_dtypes(raw_dtype, len(arrays))

    results = [
        convert_single_tensor(arr, dt, target_device, preserve_integers)
        for arr, dt in zip(arrays, target_dtypes)
    ]
    return results[0] if len(results) == 1 else tuple(results)

def to_tensor(
    *args: Any,
    target_device: Optional[str] = None,
    dtype: Union[str, type, np.dtype, Sequence[Union[str, type, np.dtype]], None] = None,
    preserve_integers: bool = True
) -> Union[Any, Tuple[Any, ...], None]:
    """Converts array-like inputs into backend tensors (NumPy or CuPy).

    Flexible API for precision handling and hardware device placement. Accepts
    individual arrays or multiple positional arrays (e.g., training/testing sets,
    feature/label splits) and casts them based on provided specifications.

    Args:
        *args: One or more array-like objects to convert. May optionally include
            a trailing positional argument specifying target `dtype`(s).
        target_device: Hardware execution backend ('numpy' or 'cupy'). If None,
            defaults to the active device specified in `aether.config`.
        dtype: Desired floating-point data type. Can be passed as a single dtype
            (broadcasted across all arrays) or a sequence/tuple matching the number
            of passed arrays 1:1.
        preserve_integers: If True, bypasses `dtype` casting for integer-based arrays
            (e.g., target labels or indices) to prevent array indexing bugs.

    Returns:
        Converted backend tensor if a single input array was provided, a tuple of
        converted backend tensors if multiple inputs were provided, or None if input
        is empty.

    Raises:
        ValueError: If `dtype` tuple length does not match the number of input arrays,
            or if an unsupported `dtype` string/type is provided.

    Warns:
        UserWarning: If `float64` precision is specified, cautioning against hardware
            performance degradation.

    Examples:
        >>> import aether as ae
        >>> X_tr, y_tr = ae.to_tensor(X_tr, y_tr, dtype='float32')
        >>> X_tr, X_ts, y_tr, y_ts = ae.to_tensor(X_tr, X_ts, y_tr, y_ts, dtype=('float32', 'float32', None, None))
    """
    arrays, raw_dtype = parse_inputs(args, dtype)
    if not arrays:
        return None

    if target_device is None:
        target_device = 'cupy' if (config.HAS_CUPY and config.xp == config.cp) else 'numpy'

    target_dtypes = resolve_dtypes(raw_dtype, len(arrays))

    results = [
        convert_single_tensor(arr, dt, target_device, preserve_integers)
        for arr, dt in zip(arrays, target_dtypes)
    ]
    return results[0] if len(results) == 1 else tuple(results)


class ToTensor:
    """Callable pipeline transform wrapper for converting inputs into backend tensors.

    Stores configuration parameters for precision and device targets so that
    tensor conversion can be consistently executed as part of an end-to-end data pipeline.

    Args:
        dtype: Desired target precision for floating-point tensors ('float16',
            'float32', etc.). Can be a single dtype or a tuple matching positional input arguments.
        preserve_integers: If True, integer-typed arrays (e.g., targets or labels)
            retain their integer precision during evaluation. Defaults to True.
        target_device: Specific execution backend target ('numpy' or 'cupy'). If None,
            defers to active environment setup in `aether.config`.
    """
    def __init__(self, dtype=None, preserve_integers=True, target_device=None):
        self.dtype = dtype
        self.preserve_integers = preserve_integers
        self.target_device = target_device

    def __call__(self, *arrays):
        """Executes tensor conversion on provided arrays using pre-configured settings.

        Args:
            *arrays: One or more array-like objects to transform.

        Returns:
            Single converted array or tuple of converted arrays.
        """
        return to_tensor(
            *arrays,
            target_device=self.target_device,
            dtype=self.dtype,
            preserve_integers=self.preserve_integers
        )

class StandardScaler:
    """
    Calculates mean and standard deviation per feature during initialization/fitting,
    and applies standardization centering to input datasets. 

    Args:
        mean: Optional pre-computed mean vector.
        std: Optional pre-computed standard deviation vector.
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
        self.dtype = None
    def fit(self, X):
        """Computes feature mean and standard deviation from dataset X.

        Args:
            X: Input dataset array used to compute normalization statistics.

        Returns:
            The fitted instance of StandardScaler.
        """
        xp = config.get_array_module(X)
        self.dtype = np.float32 if np.issubdtype(X.dtype, np.integer) else X.dtype
        
        X_float = X.astype(self.dtype, copy=False)
        self.mean = xp.mean(X_float)
        self.std = xp.std(X_float)
        return self

    def __call__(self, X):
        """Applies feature scaling to input matrix X.

        Args:
            X: Input array to scale using computed mean and standard deviation.

        Returns:
            Standardized feature tensor.

        Raises:
            ValueError: If called before `fit()` has been run or without passing `mean`/`std`
                to `__init__`.
        """
        clip = 1e-8
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler must be fit or provided mean/std before calling")
        # Ensure input array matches standard scaler dtype before op
        if hasattr(X, "astype") and X.dtype != self.dtype:
            X = X.astype(self.dtype, copy=False)
        return (X-self.mean) / (self.std + clip)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for transform in self.transforms:
            if isinstance(args, tuple):
                args = transform(*args)
            else:
                args = transform(args)
        return args