import aether.config as config

_softmax_cce_backward_kernel = config.build_kernel(
    lambda: config.cp.ElementwiseKernel(
        'float32 dvalues, int64 y_true_row, int64 class_idx, float32 smooth_offset, float32 target_offset, float32 inv_samples',
        'float32 dinputs',
        '''
        float target = (class_idx == y_true_row) ? target_offset : 0.0f;
        dinputs = (dvalues - smooth_offset - target) * inv_samples;
        ''',
        'softmax_cce_backward_kernel'
    ),
    name='_softmax_cce_backward_kernel'
)

def is_gpu_softmax_cce_backward_available() -> bool:
    """Checks if CuPy hardware support and kernels are loaded."""
    return _softmax_cce_backward_kernel is not None

def softmax_cce_backward(
        dvalues,
        y_true_row: int, 
        class_idx: int, 
        smooth_offset: float, 
        target_offset: float, 
        inv_samples: float):

    return _softmax_cce_backward_kernel(
        dvalues, 
        y_true_row, 
        class_idx, 
        smooth_offset, 
        target_offset, 
        inv_samples)