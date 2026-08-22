from string import Template
import numpy as np
import aether.config as config

# --- Fused Adam/AdamW update kernel ----------------------------------------------

_ADAMW_UPDATE_TEMPLATE = Template(r'''
$hip_include
extern "C" __global__
void $kernel_name(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ momentum,
    float* __restrict__ cache,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float bias_correction1,
    const float bias_correction2,
    const float weight_decay,
    const float l1_reg,
    const float l2_reg,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float p = param[idx];
    float g = grad[idx];

    if (l1_reg > 0.0f) {
        g += l1_reg * (p < 0.0f ? -1.0f : 1.0f);
    }
    if (l2_reg > 0.0f) {
        g += 2.0f * l2_reg * p;
    }

    float m = beta1 * momentum[idx] + (1.0f - beta1) * g;
    float v = beta2 * cache[idx] + (1.0f - beta2) * (g * g);
    momentum[idx] = m;
    cache[idx] = v;

    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;

    // Decoupled weight decay (AdamW), applied directly to the parameter
    // BEFORE the adaptive step.
    if (weight_decay > 0.0f) {
        p -= lr * weight_decay * p;
    }

    p -= lr * m_hat / (sqrtf(v_hat) + eps);

    param[idx] = p;
}
''')

_ADAMW_CUDA = dict(hip_include="")
_ADAMW_HIP = dict(hip_include="#include <hip/hip_runtime.h>\n")

_adamw_kernel_cache = {}


def _get_compiled_adamw_kernel(variant: str):
    """Cached, compiled fused AdamW RawKernel for the given vendor variant.

    Single kernel signature 'variant' ('cuda' or 'hip') is the only cache axis,
    the same memoization pattern as pooling_kernel.py's
    _get_compiled_max_backward_kernel. A failed compile is cached as None so
    a broken build doesn't re-attempt (and re-warn) on every optimizer step.
    """
    if variant in _adamw_kernel_cache:
        return _adamw_kernel_cache[variant]

    vendor = _ADAMW_HIP if variant == "hip" else _ADAMW_CUDA
    kernel_name = f"fused_adamw_update_{variant}_kernel"
    source = _ADAMW_UPDATE_TEMPLATE.substitute(
        kernel_name=kernel_name,
        **vendor,
    )

    kernel = config.build_kernel(
        lambda: config.cp.RawKernel(source, kernel_name),
        name=f"fused_adamw_update_{variant}",
    )
    _adamw_kernel_cache[variant] = kernel
    return kernel


def is_gpu_adamw_available(variant: str) -> bool:
    """Checks if CuPy hardware support and the fused kernel are loaded/compiled."""
    return _get_compiled_adamw_kernel(variant) is not None


def launch_adamw_update(
    kernel,
    param, grad, momentum, cache,
    lr, beta1, beta2, eps,
    bias_correction1, bias_correction2,
    weight_decay, l1_reg, l2_reg,
    block_size: int = 512,
):
    """Launches the fused AdamW update kernel over a flat parameter buffer.

    Assumes `param`/`grad`/`momentum`/`cache` are C-contiguous float32 arrays
    of identical shape -- true by construction at every allocation site in
    this project (Dense/Conv2d weight init)
    """
    size = param.size
    blocks = (size + block_size - 1) // block_size

    kernel(
        (blocks,), (block_size,),
        (
            param, grad, momentum, cache,
            np.float32(lr), np.float32(beta1), np.float32(beta2), np.float32(eps),
            np.float32(bias_correction1), np.float32(bias_correction2),
            np.float32(weight_decay), np.float32(l1_reg), np.float32(l2_reg),
            np.int32(size),
        ),
    )