"""
Implicit-GEMM tensor/matrix-core convolution forward kernel.
This file compiles a cupy.RawKernel per SPATIAL/CHANNEL/FILTER shape signature
using rocWMMA (AMD) / nvcuda::wmma (NVIDIA) fragments so the inner GEMM loop
maps onto Matrix Cores / Tensor Cores. 
"""
import os
import warnings
from dataclasses import dataclass
from string import Template

import aether.config as config


# --- Vendor capability probing --------------------------------------------

def _is_hip_backend():
    if not config.HAS_CUPY:
        return False
    try:
        return bool(config.cp.cuda.runtime.is_hip)
    except AttributeError:
        return False


_GPU_CONV_AVAILABLE = None  # tri-state memo: None = not probed yet


def get_is_conv_gpu_available():
    """
    memoized capability probe - checks whether the active device is
    architecturally capable of rocWMMA or nvcuda::wmma operations. Does NOT compile
    anything
    """
    global _GPU_CONV_AVAILABLE
    if _GPU_CONV_AVAILABLE is not None:
        return _GPU_CONV_AVAILABLE

    if not config.HAS_CUPY:
        _GPU_CONV_AVAILABLE = False
        return False

    try:
        device = config.cp.cuda.Device()
        if _is_hip_backend():
            props = config.cp.cuda.runtime.getDeviceProperties(device.id)
            arch = props['gcnArchName']
            arch = arch.decode() if isinstance(arch, bytes) else arch
            # gfx9xx: CDNA (MFMA-backed rocWMMA). gfx11xx: RDNA3 (WMMA-backed rocWMMA).
            _GPU_CONV_AVAILABLE = (
                arch.startswith('gfx9') or 
                arch.startswith('gfx11') or 
                arch.startswith('gfx12')
            )
        else:
            # NOTE: cupy exposes this as a numeric string, e.g. '70', '86' -
            # verify against your installed cupy version, this has shifted
            # representation across cupy releases before.
            cc = int(device.compute_capability)
            _GPU_CONV_AVAILABLE = cc >= 70
    except Exception as e:
        warnings.warn(
            f"[aether] conv_kernel: capability probe failed, disabling "
            f"matrix-core conv path: {e}"
        )
        _GPU_CONV_AVAILABLE = False

    return _GPU_CONV_AVAILABLE


# --- Shape metadata / cache key --------------------------------------------

@dataclass(frozen=True)
class ConvShapeMeta:
    H_in: int
    W_in: int
    C_in: int
    C_out: int
    fH: int
    fW: int
    sH: int
    sW: int
    pad_h: int
    pad_w: int
    H_out: int
    W_out: int
    is_hip: bool

    @property
    def K_total(self):
        return self.fH * self.fW * self.C_in

    @property
    def cache_key(self):
        # S excluded on purpose - see module docstring.
        return (
            self.H_in, self.W_in, self.C_in, self.C_out,
            self.fH, self.fW, self.sH, self.sW,
            self.pad_h, self.pad_w, self.is_hip,
        )


# Module-private tile config - not user-facing knobs (YAGNI per project
# convention). 64x64 output tile / 16x16x16 fragments is a starting point,
# not a profiled optimum; see "what needs verifying" note at the end.
_BLOCK_TILE_M = 64
_BLOCK_TILE_N = 64
_WMMA_DIM = 16


# --- Vendor substitution dictionaries --------------------------------------

_CONV_CUDA = dict(
    hip_include="#include <mma.h>\n#include <cuda_fp16.h>",
    wmma_ns="nvcuda::wmma",
    warp_size=32,
    frag_layout_a="nvcuda::wmma::row_major",
    frag_layout_b="nvcuda::wmma::row_major",
    mem_layout="nvcuda::wmma::mem_row_major",
    matrix_a_tag="nvcuda::wmma::matrix_a",
    matrix_b_tag="nvcuda::wmma::matrix_b",
    accumulator_tag="nvcuda::wmma::accumulator",
)

_CONV_HIP = dict(
    hip_include="#include <hip/hip_fp16.h>\n#include <rocwmma/rocwmma.hpp>",
    wmma_ns="rocwmma",
    warp_size=64,
    frag_layout_a="rocwmma::row_major",
    frag_layout_b="rocwmma::row_major",
    mem_layout="rocwmma::mem_row_major",
    matrix_a_tag="rocwmma::matrix_a",
    matrix_b_tag="rocwmma::matrix_b",
    accumulator_tag="rocwmma::accumulator",
)


# --- Template ---------------------------------------------------------------

CNN_FORWARD_TEMPLATE = Template(r'''
$hip_include

// ---- compile-time shape constants: baked per (H_in,W_in,C_in,C_out,fH,fW,
//      sH,sW,pad_h,pad_w) - NOT per batch size. See ConvShapeMeta.cache_key. ----
#define C_IN    $c_in
#define C_OUT   $c_out
#define F_H     $f_h
#define F_W     $f_w
#define STR_H   $str_h
#define STR_W   $str_w
#define PAD_H   $pad_h
#define PAD_W   $pad_w
#define H_IN    $h_in
#define W_IN    $w_in
#define H_OUT   $h_out
#define W_OUT   $w_out
#define K_TOTAL (F_H * F_W * C_IN)

#define WMMA_M  $wmma_dim
#define WMMA_N  $wmma_dim
#define WMMA_K  $wmma_dim

#define BLOCK_TILE_M $block_tile_m
#define BLOCK_TILE_N $block_tile_n
#define WARPS_M (BLOCK_TILE_M / WMMA_M)
#define WARPS_N (BLOCK_TILE_N / WMMA_N)
#define WARP_SIZE $warp_size


using $wmma_ns::load_matrix_sync;
using $wmma_ns::store_matrix_sync;
using $wmma_ns::mma_sync;
using $wmma_ns::fill_fragment;
using $wmma_ns::fragment;

extern "C" __global__ void conv_forward_wmma(
    const half* __restrict__ input,        // [S, H_IN, W_IN, C_IN] - UNPADDED
    const half* __restrict__ weight_fp16,  // [K_TOTAL, C_OUT], flattened (fh,fw,c_in) x c_out
    const float* __restrict__ bias,        // [C_OUT]
    float* __restrict__ output,            // [S, H_OUT, W_OUT, C_OUT]
    const long long M_total                // S * H_OUT * W_OUT - runtime, varies with batch size
) {
    const long long m_tile_start = (long long)blockIdx.x * BLOCK_TILE_M;
    const int n_tile_start = blockIdx.y * BLOCK_TILE_N;

    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;

    // +8 on the leading dim to dodge shared-memory bank conflicts on fragment loads.
    __shared__ half As[BLOCK_TILE_M][WMMA_K + 8];
    __shared__ half Bs[WMMA_K][BLOCK_TILE_N + 8];

    fragment<$matrix_a_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_a> a_frag;
    fragment<$matrix_b_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_b> b_frag;
    fragment<$accumulator_tag, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    const int num_k_tiles = (K_TOTAL + WMMA_K - 1) / WMMA_K;

    for (int kt = 0; kt < num_k_tiles; ++kt) {

        // --- A tile: implicit im2col gather. One predicate zero-fills BOTH
        //     spatial padding AND the K-dimension tail (K_TOTAL not a
        //     multiple of WMMA_K)
        for (int idx = threadIdx.x; idx < BLOCK_TILE_M * WMMA_K; idx += blockDim.x) {
            const int local_m = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const long long m_global = m_tile_start + local_m;
            const int k_global = kt * WMMA_K + local_k;

            half val = __float2half(0.0f);
            if (m_global < M_total && k_global < K_TOTAL) {
                // H_OUT, W_OUT, C_IN, F_W are compile-time constants - nvrtc/
                // hiprtc constant-fold these divisions on their own (see the
                // magic-number note above the code).
                const long long s     = m_global / (H_OUT * W_OUT);
                const long long rem_m = m_global % (H_OUT * W_OUT);
                const int h_out = (int)(rem_m / W_OUT);
                const int w_out = (int)(rem_m % W_OUT);

                const int c_in = k_global % C_IN;
                const int fw   = (k_global / C_IN) % F_W;
                const int fh   = k_global / (C_IN * F_W);

                const int h_in = h_out * STR_H - PAD_H + fh;
                const int w_in = w_out * STR_W - PAD_W + fw;

                if (h_in >= 0 && h_in < H_IN && w_in >= 0 && w_in < W_IN) {
                    const long long in_idx = ((s * H_IN + h_in) * W_IN + w_in) * C_IN + c_in;
                    val = input[in_idx];
                }
            }
            As[local_m][local_k] = val;
        }

        // --- B tile: filter weights, already cast to fp16 host-side (shadow
        //     buffer refresh - see Conv._refresh_fp16_weights). ---
        for (int idx = threadIdx.x; idx < WMMA_K * BLOCK_TILE_N; idx += blockDim.x) {
            const int local_k = idx / BLOCK_TILE_N;
            const int local_n = idx % BLOCK_TILE_N;
            const int k_global = kt * WMMA_K + local_k;
            const int n_global = n_tile_start + local_n;

            half val = __float2half(0.0f);
            if (k_global < K_TOTAL && n_global < C_OUT) {
                val = weight_fp16[(long long)k_global * C_OUT + n_global];
            }
            Bs[local_k][local_n] = val;
        }

        __syncthreads();

        if (warp_row < WARPS_M && warp_col < WARPS_N) {
            load_matrix_sync(a_frag, &As[warp_row * WMMA_M][0], WMMA_K + 8);
            load_matrix_sync(b_frag, &Bs[0][warp_col * WMMA_N], BLOCK_TILE_N + 8);
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads();
    }

    //     Bias add and fp32 store-back. Boundary-checked against the
    //     TRUE (M_total, C_OUT) tail tile 
    __shared__ float Cs[BLOCK_TILE_M][BLOCK_TILE_N];
    if (warp_row < WARPS_M && warp_col < WARPS_N) {
        store_matrix_sync(&Cs[warp_row * WMMA_M][warp_col * WMMA_N], acc_frag,
                           BLOCK_TILE_N, $mem_layout);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < BLOCK_TILE_M * BLOCK_TILE_N; idx += blockDim.x) {
        const int local_m = idx / BLOCK_TILE_N;
        const int local_n = idx % BLOCK_TILE_N;
        const long long m_global = m_tile_start + local_m;
        const int n_global = n_tile_start + local_n;

        if (m_global < M_total && n_global < C_OUT) {
            output[m_global * C_OUT + n_global] = Cs[local_m][local_n] + bias[n_global];
        }
    }
}
''')

# --- Backward Templates -----------------------------------------------------

CNN_BACKWARD_DWEIGHT_TEMPLATE = Template(r'''
$hip_include

#define C_IN    $c_in
#define C_OUT   $c_out
#define F_H     $f_h
#define F_W     $f_w
#define STR_H   $str_h
#define STR_W   $str_w
#define PAD_H   $pad_h
#define PAD_W   $pad_w
#define H_IN    $h_in
#define W_IN    $w_in
#define H_OUT   $h_out
#define W_OUT   $w_out
#define K_TOTAL (F_H * F_W * C_IN)

#define WMMA_M  $wmma_dim
#define WMMA_N  $wmma_dim
#define WMMA_K  $wmma_dim

#define BLOCK_TILE_M $block_tile_m
#define BLOCK_TILE_N $block_tile_n
#define WARPS_M (BLOCK_TILE_M / WMMA_M)
#define WARPS_N (BLOCK_TILE_N / WMMA_N)
#define WARP_SIZE $warp_size

using $wmma_ns::load_matrix_sync;
using $wmma_ns::store_matrix_sync;
using $wmma_ns::mma_sync;
using $wmma_ns::fill_fragment;
using $wmma_ns::fragment;

extern "C" __global__ void conv_backward_dweight_wmma(
    const half* __restrict__ input,        // [S, H_IN, W_IN, C_IN]
    const half* __restrict__ dvalues,      // [S, H_OUT, W_OUT, C_OUT]
    float* __restrict__ dweights,          // [K_TOTAL, C_OUT]
    const long long M_total                // S * H_OUT * W_OUT
) {
    // implicit GEMM mapping: dW = A^T [K_TOTAL, M_total] * dY [M_total, C_OUT]
    const int k_tile_start = blockIdx.x * BLOCK_TILE_M; 
    const int n_tile_start = blockIdx.y * BLOCK_TILE_N; 

    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;

    // +8 padding to dodge shared-memory bank conflicts
    __shared__ half As[BLOCK_TILE_M][WMMA_K + 8];
    __shared__ half Bs[WMMA_K][BLOCK_TILE_N + 8];

    fragment<$matrix_a_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_a> a_frag;
    fragment<$matrix_b_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_b> b_frag;
    fragment<$accumulator_tag, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    const int num_m_tiles = (M_total + WMMA_K - 1) / WMMA_K;

    for (int mt = 0; mt < num_m_tiles; ++mt) {
        
        for (int idx = threadIdx.x; idx < BLOCK_TILE_M * WMMA_K; idx += blockDim.x) {
            const int local_k = idx / WMMA_K;
            const int local_m = idx % WMMA_K;
            const int k_global = k_tile_start + local_k;
            const long long m_global = mt * WMMA_K + local_m;

            half val = __float2half(0.0f);
            if (k_global < K_TOTAL && m_global < M_total) {
                const long long s     = m_global / (H_OUT * W_OUT);
                const long long rem_m = m_global % (H_OUT * W_OUT);
                const int h_out = (int)(rem_m / W_OUT);
                const int w_out = (int)(rem_m % W_OUT);

                const int c_in = k_global % C_IN;
                const int fw   = (k_global / C_IN) % F_W;
                const int fh   = k_global / (C_IN * F_W);

                const int h_in = h_out * STR_H - PAD_H + fh;
                const int w_in = w_out * STR_W - PAD_W + fw;

                if (h_in >= 0 && h_in < H_IN && w_in >= 0 && w_in < W_IN) {
                    const long long in_idx = ((s * H_IN + h_in) * W_IN + w_in) * C_IN + c_in;
                    val = input[in_idx];
                }
            }
            As[local_k][local_m] = val;
        }

        // --- B tile: load incoming gradients dY ---
        for (int idx = threadIdx.x; idx < WMMA_K * BLOCK_TILE_N; idx += blockDim.x) {
            const int local_m = idx / BLOCK_TILE_N;
            const int local_n = idx % BLOCK_TILE_N;
            const long long m_global = mt * WMMA_K + local_m;
            const int n_global = n_tile_start + local_n;

            half val = __float2half(0.0f);
            if (m_global < M_total && n_global < C_OUT) {
                val = dvalues[m_global * C_OUT + n_global];
            }
            Bs[local_m][local_n] = val;
        }

        __syncthreads();

        if (warp_row < WARPS_M && warp_col < WARPS_N) {
            load_matrix_sync(a_frag, &As[warp_row * WMMA_M][0], WMMA_K + 8);
            load_matrix_sync(b_frag, &Bs[0][warp_col * WMMA_N], BLOCK_TILE_N + 8);
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }

    __shared__ float Cs[BLOCK_TILE_M][BLOCK_TILE_N];
    if (warp_row < WARPS_M && warp_col < WARPS_N) {
        store_matrix_sync(&Cs[warp_row * WMMA_M][warp_col * WMMA_N], acc_frag, BLOCK_TILE_N, $mem_layout);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < BLOCK_TILE_M * BLOCK_TILE_N; idx += blockDim.x) {
        const int local_k = idx / BLOCK_TILE_N;
        const int local_n = idx % BLOCK_TILE_N;
        const int k_global = k_tile_start + local_k;
        const int n_global = n_tile_start + local_n;

        if (k_global < K_TOTAL && n_global < C_OUT) {
            dweights[k_global * C_OUT + n_global] = Cs[local_k][local_n];
        }
    }
}
''')

CNN_BACKWARD_DINPUT_TEMPLATE = Template(r'''
$hip_include

#define C_IN    $c_in
#define C_OUT   $c_out
#define F_H     $f_h
#define F_W     $f_w
#define STR_H   $str_h
#define STR_W   $str_w
#define PAD_H   $pad_h
#define PAD_W   $pad_w
#define H_IN    $h_in
#define W_IN    $w_in
#define H_OUT   $h_out
#define W_OUT   $w_out
#define K_TOTAL (F_H * F_W * C_IN)

#define WMMA_M  $wmma_dim
#define WMMA_N  $wmma_dim
#define WMMA_K  $wmma_dim

#define BLOCK_TILE_M $block_tile_m
#define BLOCK_TILE_N $block_tile_n
#define WARPS_M (BLOCK_TILE_M / WMMA_M)
#define WARPS_N (BLOCK_TILE_N / WMMA_N)
#define WARP_SIZE $warp_size

using $wmma_ns::load_matrix_sync;
using $wmma_ns::store_matrix_sync;
using $wmma_ns::mma_sync;
using $wmma_ns::fill_fragment;
using $wmma_ns::fragment;

extern "C" __global__ void conv_backward_dinput_wmma(
    const half* __restrict__ dvalues,      // [S, H_OUT, W_OUT, C_OUT]
    const half* __restrict__ weight_fp16,  // [K_TOTAL, C_OUT]
    float* __restrict__ dinputs,           // [S, H_IN, W_IN, C_IN] - must be zero-initialized
    const long long M_total
) {
    // implicit GEMM mapping: dX_patches = dY [M_total, C_OUT] * W^T [C_OUT, K_TOTAL]
    const long long m_tile_start = (long long)blockIdx.x * BLOCK_TILE_M;
    const int k_tile_start = blockIdx.y * BLOCK_TILE_N;

    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;

    __shared__ half As[BLOCK_TILE_M][WMMA_K + 8];
    __shared__ half Bs[WMMA_K][BLOCK_TILE_N + 8];

    fragment<$matrix_a_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_a> a_frag;
    fragment<$matrix_b_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_b> b_frag;
    fragment<$accumulator_tag, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    const int num_c_tiles = (C_OUT + WMMA_K - 1) / WMMA_K;

    for (int ct = 0; ct < num_c_tiles; ++ct) {
        
        for (int idx = threadIdx.x; idx < BLOCK_TILE_M * WMMA_K; idx += blockDim.x) {
            const int local_m = idx / WMMA_K;
            const int local_c = idx % WMMA_K;
            const long long m_global = m_tile_start + local_m;
            const int c_global = ct * WMMA_K + local_c;

            half val = __float2half(0.0f);
            if (m_global < M_total && c_global < C_OUT) {
                val = dvalues[m_global * C_OUT + c_global];
            }
            As[local_m][local_c] = val;
        }

        for (int idx = threadIdx.x; idx < WMMA_K * BLOCK_TILE_N; idx += blockDim.x) {
            const int local_c = idx / BLOCK_TILE_N;
            const int local_k = idx % BLOCK_TILE_N;
            const int c_global = ct * WMMA_K + local_c;
            const int k_global = k_tile_start + local_k;

            half val = __float2half(0.0f);
            if (c_global < C_OUT && k_global < K_TOTAL) {
                // Notice the access pattern to weight_fp16: [k_global, c_global]
                val = weight_fp16[(long long)k_global * C_OUT + c_global];
            }
            Bs[local_c][local_k] = val;
        }

        __syncthreads();

        if (warp_row < WARPS_M && warp_col < WARPS_N) {
            load_matrix_sync(a_frag, &As[warp_row * WMMA_M][0], WMMA_K + 8);
            load_matrix_sync(b_frag, &Bs[0][warp_col * WMMA_N], BLOCK_TILE_N + 8);
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }

    __shared__ float Cs[BLOCK_TILE_M][BLOCK_TILE_N];
    if (warp_row < WARPS_M && warp_col < WARPS_N) {
        store_matrix_sync(&Cs[warp_row * WMMA_M][warp_col * WMMA_N], acc_frag, BLOCK_TILE_N, $mem_layout);
    }
    __syncthreads();

    // --- Epilogue: Fused Atomic Scatter-Add (Col2Im) ---
    for (int idx = threadIdx.x; idx < BLOCK_TILE_M * BLOCK_TILE_N; idx += blockDim.x) {
        const int local_m = idx / BLOCK_TILE_N;
        const int local_k = idx % BLOCK_TILE_N;
        const long long m_global = m_tile_start + local_m;
        const int k_global = k_tile_start + local_k;

        if (m_global < M_total && k_global < K_TOTAL) {
            const long long s     = m_global / (H_OUT * W_OUT);
            const long long rem_m = m_global % (H_OUT * W_OUT);
            const int h_out = (int)(rem_m / W_OUT);
            const int w_out = (int)(rem_m % W_OUT);

            const int c_in = k_global % C_IN;
            const int fw   = (k_global / C_IN) % F_W;
            const int fh   = k_global / (C_IN * F_W);

            const int h_in = h_out * STR_H - PAD_H + fh;
            const int w_in = w_out * STR_W - PAD_W + fw;

            if (h_in >= 0 && h_in < H_IN && w_in >= 0 && w_in < W_IN) {
                const long long in_idx = ((s * H_IN + h_in) * W_IN + w_in) * C_IN + c_in;
                atomicAdd(&dinputs[in_idx], Cs[local_m][local_k]);
            }
        }
    }
}
''')

CNN_BACKWARD_DBIAS_TEMPLATE = Template(r'''
$hip_include

extern "C" __global__ void conv_backward_dbias(
    const float* __restrict__ dvalues, // [S * H_OUT * W_OUT, C_OUT]
    float* __restrict__ dbiases,       // [C_OUT]
    const long long M_total,
    const int C_OUT
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C_OUT) {
        float sum = 0.0f;
        for (long long m = 0; m < M_total; ++m) {
            sum += dvalues[m * C_OUT + c];
        }
        dbiases[c] = sum;
    }
}
''')
# --- Forward Compilation + caching --------------------------------------------------

_CONV_FORWARD_KERNEL_CACHE = {}


def get_compiled_forward_conv_kernel(shape_meta: ConvShapeMeta):
    """
    Cached, compiled RawKernel for one shape signature (batch size excluded).
    A failed compile is cached as None so a broken shape doesn't re-attempt 
    (and re-warn) every forward call
    """
    key = shape_meta.cache_key
    if key in _CONV_FORWARD_KERNEL_CACHE:
        return _CONV_FORWARD_KERNEL_CACHE[key]

    vendor = _CONV_HIP if shape_meta.is_hip else _CONV_CUDA

    source = CNN_FORWARD_TEMPLATE.substitute(
        c_in=shape_meta.C_in, c_out=shape_meta.C_out,
        f_h=shape_meta.fH, f_w=shape_meta.fW,
        str_h=shape_meta.sH, str_w=shape_meta.sW,
        pad_h=shape_meta.pad_h, pad_w=shape_meta.pad_w,
        h_in=shape_meta.H_in, w_in=shape_meta.W_in,
        h_out=shape_meta.H_out, w_out=shape_meta.W_out,
        wmma_dim=_WMMA_DIM,
        block_tile_m=_BLOCK_TILE_M, block_tile_n=_BLOCK_TILE_N,
        **vendor,
    )

    options = ('-std=c++17',)
    if shape_meta.is_hip:
        # rocWMMA is a separate header-only lib from base ROCm/HIP - it must
        # be locatable by hiprtc. This is a real v1.0 packaging item beyond
        # the numpy/cupy split already tracked for roadmap step 9.
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        options = options + (f'-I{rocm_path}/include',)

    try:
        module = config.cp.RawModule(code=source, options=options)
        kernel = module.get_function('conv_forward_wmma')
    except Exception as e:
        warnings.warn(
            f"[aether] conv_kernel: matrix-core conv compile failed for "
            f"shape {key}, this Conv layer will fall back to the scalar "
            f"path: {e}"
        )
        _CONV_FORWARD_KERNEL_CACHE[key] = None
        return None

    threads_per_block = (_BLOCK_TILE_M // _WMMA_DIM) * (_BLOCK_TILE_N // _WMMA_DIM) * vendor['warp_size']
    launch_meta = dict(
        block=(threads_per_block,),
        block_tile_m=_BLOCK_TILE_M,
        block_tile_n=_BLOCK_TILE_N,
    )

    _CONV_FORWARD_KERNEL_CACHE[key] = (kernel, launch_meta)
    return _CONV_FORWARD_KERNEL_CACHE[key]

# --- Backward Compilation + caching -----------------------------------------

_CONV_BACKWARD_DWEIGHT_CACHE = {}
_CONV_BACKWARD_DINPUT_CACHE = {}
_CONV_BACKWARD_DBIAS_CACHE = {}


def get_compiled_backward_dweight_kernel(shape_meta: ConvShapeMeta):
    key = shape_meta.cache_key
    if key in _CONV_BACKWARD_DWEIGHT_CACHE:
        return _CONV_BACKWARD_DWEIGHT_CACHE[key]

    vendor = _CONV_HIP if shape_meta.is_hip else _CONV_CUDA
    source = CNN_BACKWARD_DWEIGHT_TEMPLATE.substitute(
        c_in=shape_meta.C_in, c_out=shape_meta.C_out,
        f_h=shape_meta.fH, f_w=shape_meta.fW,
        str_h=shape_meta.sH, str_w=shape_meta.sW,
        pad_h=shape_meta.pad_h, pad_w=shape_meta.pad_w,
        h_in=shape_meta.H_in, w_in=shape_meta.W_in,
        h_out=shape_meta.H_out, w_out=shape_meta.W_out,
        wmma_dim=_WMMA_DIM,
        block_tile_m=_BLOCK_TILE_M, block_tile_n=_BLOCK_TILE_N,
        **vendor,
    )

    options = ('-std=c++17',)
    if shape_meta.is_hip:
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        options = options + (f'-I{rocm_path}/include',)

    try:
        module = config.cp.RawModule(code=source, options=options)
        kernel = module.get_function('conv_backward_dweight_wmma')
    except Exception as e:
        warnings.warn(f"[aether] conv_kernel: matrix-core dweight compile failed: {e}")
        _CONV_BACKWARD_DWEIGHT_CACHE[key] = None
        return None

    threads_per_block = (_BLOCK_TILE_M // _WMMA_DIM) * (_BLOCK_TILE_N // _WMMA_DIM) * vendor['warp_size']
    launch_meta = dict(
        block=(threads_per_block,),
        block_tile_m=_BLOCK_TILE_M,
        block_tile_n=_BLOCK_TILE_N,
    )

    _CONV_BACKWARD_DWEIGHT_CACHE[key] = (kernel, launch_meta)
    return _CONV_BACKWARD_DWEIGHT_CACHE[key]


def get_compiled_backward_dinput_kernel(shape_meta: ConvShapeMeta):
    key = shape_meta.cache_key
    if key in _CONV_BACKWARD_DINPUT_CACHE:
        return _CONV_BACKWARD_DINPUT_CACHE[key]

    vendor = _CONV_HIP if shape_meta.is_hip else _CONV_CUDA
    source = CNN_BACKWARD_DINPUT_TEMPLATE.substitute(
        c_in=shape_meta.C_in, c_out=shape_meta.C_out,
        f_h=shape_meta.fH, f_w=shape_meta.fW,
        str_h=shape_meta.sH, str_w=shape_meta.sW,
        pad_h=shape_meta.pad_h, pad_w=shape_meta.pad_w,
        h_in=shape_meta.H_in, w_in=shape_meta.W_in,
        h_out=shape_meta.H_out, w_out=shape_meta.W_out,
        wmma_dim=_WMMA_DIM,
        block_tile_m=_BLOCK_TILE_M, block_tile_n=_BLOCK_TILE_N,
        **vendor,
    )

    options = ('-std=c++17',)
    if shape_meta.is_hip:
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        options = options + (f'-I{rocm_path}/include',)

    try:
        module = config.cp.RawModule(code=source, options=options)
        kernel = module.get_function('conv_backward_dinput_wmma')
    except Exception as e:
        warnings.warn(f"[aether] conv_kernel: matrix-core dinput compile failed: {e}")
        _CONV_BACKWARD_DINPUT_CACHE[key] = None
        return None

    threads_per_block = (_BLOCK_TILE_M // _WMMA_DIM) * (_BLOCK_TILE_N // _WMMA_DIM) * vendor['warp_size']
    launch_meta = dict(
        block=(threads_per_block,),
        block_tile_m=_BLOCK_TILE_M,
        block_tile_n=_BLOCK_TILE_N,
    )

    _CONV_BACKWARD_DINPUT_CACHE[key] = (kernel, launch_meta)
    return _CONV_BACKWARD_DINPUT_CACHE[key]


def get_compiled_backward_dbias_kernel(shape_meta: ConvShapeMeta):
    # dbias only scales with C_out, but we'll reuse the same cache_key pattern for consistency
    key = shape_meta.cache_key
    if key in _CONV_BACKWARD_DBIAS_CACHE:
        return _CONV_BACKWARD_DBIAS_CACHE[key]

    vendor = _CONV_HIP if shape_meta.is_hip else _CONV_CUDA
    source = CNN_BACKWARD_DBIAS_TEMPLATE.substitute(**vendor)

    options = ('-std=c++17',)
    if shape_meta.is_hip:
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        options = options + (f'-I{rocm_path}/include',)

    try:
        module = config.cp.RawModule(code=source, options=options)
        kernel = module.get_function('conv_backward_dbias')
    except Exception as e:
        warnings.warn(f"[aether] conv_kernel: scalar dbias compile failed: {e}")
        _CONV_BACKWARD_DBIAS_CACHE[key] = None
        return None

    launch_meta = dict(
        block=(256,),
        # Grid dimension should be calculated host-side as ceil(C_out / 256)
    )

    _CONV_BACKWARD_DBIAS_CACHE[key] = (kernel, launch_meta)
    return _CONV_BACKWARD_DBIAS_CACHE[key]

