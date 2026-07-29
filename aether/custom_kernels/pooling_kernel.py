from string import Template
import aether.config as config

_POOL_FORWARD_TEMPLATE = Template(r'''
$hip_include
extern "C" __global__
void $kernel_name(
    const float* __restrict__ x,
    float* __restrict__ out,
    $aux_out_decl
    const int S, const int H_in, const int W_in, const int C,
    const int fH, const int fW, const int sH, const int sW,
    const int H_out, const int W_out
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int w_out = blockIdx.y * blockDim.y + threadIdx.y;
    int h_s = blockIdx.z * blockDim.z + threadIdx.z;

    int h_out = h_s % H_out;
    int s = h_s / H_out;

    if (c >= C || w_out >= W_out || s >= S) return;

    int h_start = h_out * sH;
    int w_start = w_out * sW;
    int batch_offset = s * H_in * W_in * C;

    $reduce_init

    for (int fh = 0; fh < fH; ++fh) {
        int h_in = h_start + fh;
        int row_offset = batch_offset + (h_in * W_in) * C;

        for (int fw = 0; fw < fW; ++fw) {
            int w_in = w_start + fw;
            int in_idx = row_offset + w_in * C + c;

            $reduce_body
        }
    }

    int out_idx = ((s * H_out + h_out) * W_out + w_out) * C + c;
    $writeback
}
''')

_POOL_BACKWARD_NONOVERLAP_TEMPLATE = Template(r'''
$hip_include
extern "C" __global__
void $kernel_name(
    const float* __restrict__ dvalues,
    $aux_in_decl
    float* __restrict__ dinputs,
    const int S, const int H_pad, const int W_pad, const int C,
    const int fH, const int fW, const int sH, const int sW,
    const int H_out, const int W_out
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int w_out = blockIdx.y * blockDim.y + threadIdx.y;
    int h_s = blockIdx.z * blockDim.z + threadIdx.z;

    int h_out = h_s % H_out;
    int s = h_s / H_out;

    if (c >= C || w_out >= W_out || s >= S) return;

    int h_start = h_out * sH;
    int w_start = w_out * sW;
    int batch_offset = s * H_pad * W_pad * C;

    int out_idx = ((s * H_out + h_out) * W_out + w_out) * C + c;
    float dval = dvalues[out_idx];

    $reduce_init

    for (int fh = 0; fh < fH; ++fh) {
        int h_in = h_start + fh;
        int row_offset = batch_offset + (h_in * W_pad) * C;

        for (int fw = 0; fw < fW; ++fw) {
            int w_in = w_start + fw;
            int in_idx = row_offset + w_in * C + c;

            $reduce_body
        }
    }
}
''')

_MAX_OP = {
    "name": "max",
    "aux_out_decl": "int* __restrict__ max_indices,",
    "reduce_init": (
        "int initial_idx = batch_offset + (h_start * W_in + w_start) * C + c;\n"
        "    float max_val = x[initial_idx];\n"
        "    int best_idx = initial_idx;"
    ),
    "reduce_body": (
        "float val = x[in_idx];\n"
        "            if (val > max_val) { max_val = val; best_idx = in_idx; }"
    ),
    "writeback": "out[out_idx] = max_val;\n    max_indices[out_idx] = best_idx;",
}
_MAX_OP_INFERENCE = {
    "name": "max_inference",
    "aux_out_decl": "",
    "reduce_init": (
        "int initial_idx = batch_offset + (h_start * W_in + w_start) * C + c;\n"
        "    float max_val = x[initial_idx];"
    ),
    "reduce_body": (
        "float val = x[in_idx];\n"
        "            if (val > max_val) { max_val = val; }"
    ),
    "writeback": "out[out_idx] = max_val;",
}

# Only the argmax slot (recorded during forward) receives the upstream
# gradient; every other slot in the window gets nothing. Since windows
# don't overlap, that argmax slot is never revisited by another thread,
# so a direct conditional write is race-free.
_MAX_BACKWARD_NONOVERLAP_OP = {
    "name": "max_backward_nonoverlap",
    "aux_in_decl": "const int* __restrict__ max_indices,",
    "reduce_init": "int src_idx = max_indices[out_idx];",
    "reduce_body": "if (in_idx == src_idx) { dinputs[in_idx] = dval; }",
}

# Every slot in the window receives an equal share of the upstream
# gradient. No aux input needed (no argmax bookkeeping for avg-pool).
# Included now for when AvgPool2d lands; not wired up to a public
# getter yet since AvgPool2d itself doesn't exist.
_AVG_BACKWARD_NONOVERLAP_OP = {
    "name": "avg_backward_nonoverlap",
    "aux_in_decl": "",
    "reduce_init": "float avg_val = dval / (fH * fW);",
    "reduce_body": "dinputs[in_idx] = avg_val;",
}

_pool_kernel_cache = {}

def _get_compiled_forward_kernel(op_dict: dict, variant: str):
    """Retrieves or compiles a GPU pooling RawKernel for the specified backend.

    Utilizes Just-In-Time (JIT) string metaprogramming and memoization. On first
    invocation for a given operation and variant, it dynamically injects 
    operation-specific reduction logic and target platform headers into 
    `_POOL_FORWARD_TEMPLATE`, compiles the raw C++ code via CuPy/NVRTC/HIPRTC, 
    and caches the resulting executable object. Subsequent calls fetch the 
    pre-compiled kernel directly from `_pool_kernel_cache`.

    Args:
        op_dict (dict): Descriptor containing operation-specific C++ code fragments.
            Expected keys:
                - "name" (str): Operation identifier (e.g., 'max', 'avg').
                - "aux_out_decl" (str): C++ declaration for auxiliary output 
                  buffers (e.g., max index pointers).
                - "reduce_init" (str): C++ setup code prior to the window loop.
                - "reduce_body" (str): C++ logic executed for each window element.
                - "writeback" (str): C++ code to commit final values to global memory.
        variant (str): Target GPU platform architecture. Must be either 'cuda' or 'hip'.

    Returns:
        RawKernel: A compiled CuPy RawKernel instance ready to be launched 
        with grid and block dimensions.
    """

    cache_key = (op_dict["name"], variant)
    if cache_key in _pool_kernel_cache:
        return _pool_kernel_cache[cache_key]

    kernel_name = f"pool2d_forward_{op_dict['name']}_{variant}_kernel"
    source = _POOL_FORWARD_TEMPLATE.substitute(
        hip_include="#include <hip/hip_runtime.h>\n" if variant == "hip" else "",
        kernel_name=kernel_name,
        aux_out_decl=op_dict["aux_out_decl"],
        reduce_init=op_dict["reduce_init"],
        reduce_body=op_dict["reduce_body"],
        writeback=op_dict["writeback"],
    )

    kernel = config.build_kernel(
        lambda: config.cp.RawKernel(source, kernel_name),
        name=f"pool2d_forward_{op_dict['name']}_{variant}",
    )
    _pool_kernel_cache[cache_key] = kernel
    return kernel

def _get_compiled_backward_kernel(op_dict: dict, variant: str):
    """Retrieves or compiles a GPU pooling backward RawKernel for the
    non-overlapping-window (filter_size == stride) case. Mirrors
    `_get_compiled_kernel`'s memoized JIT-compile pattern, but substitutes
    into `_POOL_BACKWARD_NONOVERLAP_TEMPLATE`, which is agnostic to the
    specific pooling op — max-pool and avg-pool (and anything else that
    is a per-window reduction) plug in via `aux_in_decl` / `reduce_init` /
    `reduce_body` the same way forward ops plug into `_POOL_FORWARD_TEMPLATE`.

    Args:
        op_dict (dict): Descriptor containing operation-specific C++ code fragments.
            Expected keys:
                - "name" (str): Operation identifier (e.g., 'max_backward_nonoverlap').
                - "aux_in_decl" (str): C++ declaration for extra input buffers
                  needed by backward (e.g. max_indices for max-pool; empty for avg-pool).
                - "reduce_init" (str): C++ setup code prior to the window loop
                  (e.g. loading the argmax source index, or precomputing the
                  averaged gradient value).
                - "reduce_body" (str): C++ logic executed for each window element,
                  responsible for writing into `dinputs` directly.
        variant (str): Target GPU platform architecture. Must be either 'cuda' or 'hip'.

    Returns:
        RawKernel: A compiled CuPy RawKernel instance ready to be launched
        with grid and block dimensions.
    """

    cache_key = (op_dict["name"], variant)
    if cache_key in _pool_kernel_cache:
        return _pool_kernel_cache[cache_key]

    kernel_name = f"pool2d_backward_{op_dict['name']}_{variant}_kernel"
    source = _POOL_BACKWARD_NONOVERLAP_TEMPLATE.substitute(
        hip_include="#include <hip/hip_runtime.h>\n" if variant == "hip" else "",
        kernel_name=kernel_name,
        aux_in_decl=op_dict["aux_in_decl"],
        reduce_init=op_dict["reduce_init"],
        reduce_body=op_dict["reduce_body"],
    )

    kernel = config.build_kernel(
        lambda: config.cp.RawKernel(source, kernel_name),
        name=f"pool2d_backward_{op_dict['name']}_{variant}",
    )
    _pool_kernel_cache[cache_key] = kernel
    return kernel

# Helper Functions for MaxPool2D
def is_gpu_max_pool2d_available() -> bool:
    """Checks if CuPy hardware support and kernels are loaded."""
    return config.HAS_CUPY

def get_max_pool2d_forward_kernel(variant: str, training: bool = True):
    """Returns a compiled RawKernel for the given variant ('cuda' or 'hip').
    Memoized via _pool_kernel_cache — no lazy self-rewriting needed.
    If we're not training, remove the 4D max_indicies tensor using a similar
    kernel _MAX_OP_INFERENCE
    """
    op = _MAX_OP if training else _MAX_OP_INFERENCE
    return _get_compiled_forward_kernel(op, variant)

def get_max_pool2d_backward_kernel(variant: str):
    """Returns a compiled RawKernel implementing the MaxPool2d backward
    scatter for the non-overlapping-window case (filter_size == stride
    only). Every padded-input position is the argmax target of at most
    one output window in this regime, so the kernel does a direct write
    instead of an atomicAdd. Do NOT use this for overlapping windows —
    that requires the atomic/scatter-add fallback path instead.
    """
    return _get_compiled_backward_kernel(_MAX_BACKWARD_NONOVERLAP_OP, variant)
