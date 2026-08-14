import numpy as np
from typing import Sequence, Union, Tuple, Optional, Any
import aether.config as config
from ._utils import parse_inputs, resolve_dtypes, convert_single_tensor

ALLOWED_DTYPES = {'float16', 'bfloat16', 'float32', 'float64'}

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
    is_sequence = isinstance(raw_dtype, (tuple,list))

    results = []
    for i, (arr, dt) in enumerate(zip(arrays, target_dtypes)):
        if arr is None:
            results.append(None)
            continue
        try:
            temp_tensor = config.to_device(arr, target=target_device)
        except RuntimeError:
            temp_tensor = np.asarray(arr)

        if (
            len(arrays) > 1
            and i > 0
            and preserve_integers
            and not is_sequence
            and hasattr(temp_tensor, 'dtype')
            and temp_tensor.dtype.kind in ('i','u')
        ):
            effective_dtype = None
        else:
            effective_dtype = dt

        results.append(convert_single_tensor(arr, effective_dtype, target_device, preserve_integers))
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
        axis: Axis or axes along which statistics are computed
            -None: Scaler normalaztion across all axes
            - 0: Feature-wise normalaztion across 2D tabular data (S, D).
            - (0, 1, 2): Channel-wise normalization for SHWC images (S, H, W, C):
    """
    def __init__(self, mean=None, std=None, axis = None):
        self.mean = mean
        self.std = std
        self.axis = axis
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
        self.mean = xp.mean(X_float, axis=self.axis, keepdims=True)
        self.std = xp.std(X_float, axis=self.axis, keepdims=True)
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
        xp = config.get_array_module(X)

        if config.get_array_module(self.mean) != xp:
            self.mean = xp.asarray(self.mean)
            self.std = xp.asarray(self.std)
        # Ensure input array matches standard scaler dtype before op
        if hasattr(X, "astype") and X.dtype != self.dtype:
            X = X.astype(self.dtype, copy=False)
        return (X-self.mean) / (self.std + clip)

class Rescale:
    """Scales array inputs by a multiplier factor (e.g., 1/255.0 for sRGB uint8 images)."""
    def __init__(self, factor: float = 1.0 / 255.0):
        self.factor = factor

    def __call__(self, *args):
        if not args:
            return None
        
        if len(args) == 1:
            return args[0] * self.factor
        
        # If multiple inputs (e.g., X, y) are passed through Compose, 
        # only scale floating-point feature tensors and leave integer labels untouched.
        results = []
        for arr in args:
            if hasattr(arr, 'dtype') and arr.dtype.kind in ('f',):
                results.append(arr * self.factor)
            else:
                results.append(arr)
        return tuple(results)
    
class Compose:
    """Sequentially chains multiple data transformations into an end-to-end preprocessing pipeline.

    Executes a list of transformations in sequential order, passing the output
    of each transform as the input to the next. Supports both stateless transforms
    (e.g., `ToTensor`, `Rescale`) and stateful transforms (e.g., `StandardScaler`).

    Args:
        transforms (Sequence[Callable[..., Any]]): An ordered sequence or list of
            callable transform instances to execute.

    Attributes:
        transforms (Sequence[Callable[..., Any]]): The sequence of chained transforms.

    Examples:
        >>> import aether as ae
        >>> import numpy as np
        >>>
        >>> # Prepare raw input arrays (e.g., uint8 image batches)
        >>> X_train = np.random.randint(0, 256, size=(50000, 32, 32, 3), dtype=np.uint8)
        >>> X_test = np.random.randint(0, 256, size=(10000, 32, 32, 3), dtype=np.uint8)
        >>>
        >>> # Build and fit pipeline on training data 
        >>> TARGET_DEVICE = "cupy"
        >>> feature_pipeline = ae.Compose([
        ...     ae.ToTensor(dtype='float32', target_device=TARGET_DEVICE),
        ...     ae.Rescale(factor=1.0 / 255.0),
        ...     ae.StandardScaler()
        ... ]).fit(X_train)
        >>>
        >>> # Transform training and test partitions using fitted statistics
        >>> X_train_tensor = feature_pipeline(X_train)
        >>> X_test_tensor = feature_pipeline(X_test)
        >>>
        >>> # Alternatively, use fit_transform for a single-pass workflow
        >>> X_train_tensor = feature_pipeline.fit_transform(X_train)
    """
    def __init__(self, transforms: Sequence[Any]):
        self.transforms = transforms

    def fit_transform(self, *args: Any):
        """Fits stateful transforms and returns the final transformed result

        in a single pass.
        """
        res = args[0] if len(args) == 1 else args

        for transform in self.transforms:
            # Check for a dedicated fit_transform method on the individual transform
            if hasattr(transform, "fit_transform") and callable(transform.fit_transform):
                res = transform.fit_transform(*res) if isinstance(res, tuple) else transform.fit_transform(res)
            else:
                # Otherwise fit if stateful, then transform
                if hasattr(transform, "fit") and callable(transform.fit):
                    if isinstance(res, tuple):
                        transform.fit(*res)
                    else:
                        transform.fit(res)

                if callable(transform):
                    res = transform(*res) if isinstance(res, tuple) else transform(res)

        return res

    def fit(self, *args: Any):
        """Fits all stateful transforms in the pipeline sequentially and returns self."""
        self.fit_transform(*args)
        return self
    
    def __call__(self, *args):

        res = args[0] if len(args) == 1 else args 
        for transform in self.transforms:
            if isinstance(res, tuple):
                res = transform(*res)
            else:
                res = transform(res)
        return res