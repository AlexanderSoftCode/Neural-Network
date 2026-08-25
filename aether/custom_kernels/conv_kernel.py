import os
import warnings

import numpy as np
from dataclasses import dataclass
from string import Template

import aether.config as config


def get_is_conv_gpu_available():
    """Check if GPU supports optimized tensor-core convolution kernels."""
    return config.get_tensor_core_capable()


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
        return (
            self.H_in, self.W_in, self.C_in, self.C_out,
            self.fH, self.fW, self.sH, self.sW,
            self.pad_h, self.pad_w, self.is_hip,
        )

    @classmethod
    def create(cls, spatial_shape, C_out, filter_size, stride, padding, is_hip=None):
        """Derive padding and output extents from a layer's configuration.

        This is the only place the conv spatial geometry is computed. Callers
        -- layers, benchmarks, profilers -- must go through here rather than
        rederiving it, so a script can never silently disagree with the
        kernel about what shape it is running.
        """
        H_in, W_in, C_in = spatial_shape
        fH, fW = filter_size
        sH, sW = stride

        if padding == "same":
            pad_h, pad_w = (fH - 1) // 2, (fW - 1) // 2
        elif padding == "valid":
            pad_h, pad_w = 0, 0
        else:
            raise ValueError(
                f"Unsupported padding mode '{padding}'. Use 'same' or 'valid'."
            )

        return cls(
            H_in=H_in, W_in=W_in, C_in=C_in, C_out=C_out,
            fH=fH, fW=fW, sH=sH, sW=sW, pad_h=pad_h, pad_w=pad_w,
            H_out=int((H_in + 2 * pad_h - fH) / sH + 1),
            W_out=int((W_in + 2 * pad_w - fW) / sW + 1),
            is_hip=config.IS_HIP if is_hip is None else is_hip,
        )


_BLOCK_TILE_M = 64
_BLOCK_TILE_N = 64
_WMMA_DIM = 16
_WARP_SIZE = 32

# LDS padding for the dweight kernel's staging buffers, in halves. Tuned
# empirically against the column-major A-fragment read pattern via
# examples/cifar10/dweight_bench.py -- see the sweep there before changing.
_DWEIGHT_A_PAD = 8
_DWEIGHT_B_PAD = 8


_MP_COUNT = None


def get_device_multiprocessor_count():
    """Memoized MP count, used to size the dweight split-K grid."""
    global _MP_COUNT
    if _MP_COUNT is None:
        try:
            device = config.cp.cuda.Device()
            props = config.cp.cuda.runtime.getDeviceProperties(device.id)
            _MP_COUNT = int(props['multiProcessorCount'])
        except Exception:
            _MP_COUNT = 32
    return _MP_COUNT


def choose_dweight_split_k(num_m_tiles, grid_x, grid_y):
    """Pick the number of split-K slices for the dweight grid.

    The dweight kernel is latency-bound, not matrix-core bound: its inner loop
    stages one 16-deep k-step into LDS behind two block barriers, so a single
    block spends ~1us per iteration and the only way to cover that is to have
    many blocks in flight. Split-K is therefore the primary throughput knob.

    It is not free, though -- each extra slice repeats the whole BLOCK_TILE_M x
    BLOCK_TILE_N epilogue and its atomicAdds, so past the point where the
    machine is full, more slices strictly cost. Measured on the CIFAR shapes
    (examples/cifar10/dweight_bench.py), the optimum sits near 2 blocks per MP
    when the 2D grid already supplies blocks, and near 4 per MP when it does
    not -- with a single output tile, split-K is the *only* parallelism.
    """
    mp_count = get_device_multiprocessor_count()
    b_2d = max(1, grid_x * grid_y)
    target_blocks = (4 if b_2d == 1 else 2) * mp_count
    return max(1, min(num_m_tiles, target_blocks // b_2d))


# --- Vendor substitution dictionaries --------------------------------------

_CONV_CUDA = dict(
    hip_include="#include <mma.h>\n#include <cuda_fp16.h>",
    wmma_ns="nvcuda::wmma",
    warp_size=_WARP_SIZE,
    frag_layout_a="nvcuda::wmma::row_major",
    frag_layout_a_col="nvcuda::wmma::col_major",
    frag_layout_b="nvcuda::wmma::row_major",
    mem_layout="nvcuda::wmma::mem_row_major",
    matrix_a_tag="nvcuda::wmma::matrix_a",
    matrix_b_tag="nvcuda::wmma::matrix_b",
    accumulator_tag="nvcuda::wmma::accumulator",
)

_CONV_HIP = dict(
    hip_include="#include <hip/hip_fp16.h>\n#include <rocwmma/rocwmma.hpp>",
    wmma_ns="rocwmma",
    warp_size=_WARP_SIZE,
    frag_layout_a="rocwmma::row_major",
    frag_layout_a_col="rocwmma::col_major",
    frag_layout_b="rocwmma::row_major",
    mem_layout="rocwmma::mem_row_major",
    matrix_a_tag="rocwmma::matrix_a",
    matrix_b_tag="rocwmma::matrix_b",
    accumulator_tag="rocwmma::accumulator",
)


# --- Forward Template -------------------------------------------------------

CNN_FORWARD_TEMPLATE = Template(r'''
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
#define HW_OUT  (H_OUT * W_OUT)
#define C_IN_FW (C_IN * F_W)

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
    const half* __restrict__ input,        // [S, H_IN, W_IN, C_IN]
    const half* __restrict__ weight_fp16,  // [K_TOTAL, C_OUT]
    const float* __restrict__ bias,        // [C_OUT]
    float* __restrict__ output,            // [S, H_OUT, W_OUT, C_OUT]
    const long long M_total                // S * H_OUT * W_OUT
) {
    const long long m_tile_start = (long long)blockIdx.x * BLOCK_TILE_M;
    const int n_tile_start = blockIdx.y * BLOCK_TILE_N;

    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;

    __shared__ half As[BLOCK_TILE_M][WMMA_K + 8];
    __shared__ half Bs[WMMA_K][BLOCK_TILE_N + 8];

    fragment<$matrix_a_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_a> a_frag;
    fragment<$matrix_b_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_b> b_frag;
    fragment<$accumulator_tag, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    const int num_k_tiles = (K_TOTAL + WMMA_K - 1) / WMMA_K;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        for (int idx = threadIdx.x; idx < BLOCK_TILE_M * WMMA_K; idx += blockDim.x) {
            const int local_m = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const unsigned int m_global = (unsigned int)(m_tile_start + local_m);
            const int k_global = kt * WMMA_K + local_k;

            half val = __float2half(0.0f);
            if (m_global < (unsigned int)M_total && k_global < K_TOTAL) {
                const unsigned int s     = m_global / HW_OUT;
                const unsigned int rem_m = m_global % HW_OUT;
                const int h_out = (int)(rem_m / W_OUT);
                const int w_out = (int)(rem_m % W_OUT);

                const int c_in = k_global % C_IN;
                const int fw   = (k_global / C_IN) % F_W;
                const int fh   = k_global / C_IN_FW;

                const int h_in = h_out * STR_H - PAD_H + fh;
                const int w_in = w_out * STR_W - PAD_W + fw;

                if (h_in >= 0 && h_in < H_IN && w_in >= 0 && w_in < W_IN) {
                    const long long in_idx = (((long long)s * H_IN + h_in) * W_IN + w_in) * C_IN + c_in;
                    val = input[in_idx];
                }
            }
            As[local_m][local_k] = val;
        }

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
#define HW_OUT  (H_OUT * W_OUT)
#define C_IN_FW (C_IN * F_W)

#define WMMA_M  $wmma_dim
#define WMMA_N  $wmma_dim
#define WMMA_K  $wmma_dim

#define BLOCK_TILE_M $block_tile_m
#define BLOCK_TILE_N $block_tile_n
#define WARPS_M (BLOCK_TILE_M / WMMA_M)
#define WARPS_N (BLOCK_TILE_N / WMMA_N)
#define WARP_SIZE $warp_size
#define THREADS (WARPS_M * WARPS_N * WARP_SIZE)

// LDS padding. As is indexed [pixel][filter] so its leading dimension is
// BLOCK_TILE_M; A_PAD spreads the column-major fragment reads across banks.
#define A_PAD $a_pad
#define B_PAD $b_pad
#define A_LD (BLOCK_TILE_M + A_PAD)

using $wmma_ns::load_matrix_sync;
using $wmma_ns::store_matrix_sync;
using $wmma_ns::mma_sync;
using $wmma_ns::fill_fragment;
using $wmma_ns::fragment;

// bias_acc is sized by the B-tile strided-loop trip count; keep them in sync.
#define BIAS_ITEMS (((WMMA_K * BLOCK_TILE_N) + THREADS - 1) / THREADS)

// The A-gather hoists its filter-index decomposition out of the reduction
// loop, which is only valid if a lane's filter index never changes as the
// strided staging loop advances by THREADS.
static_assert(THREADS % BLOCK_TILE_M == 0, "THREADS must be a multiple of BLOCK_TILE_M");
static_assert(THREADS >= BLOCK_TILE_N, "bias flush assumes one thread per N column");

extern "C" __global__ void __launch_bounds__(THREADS) conv_backward_dweight_wmma(
    const half* __restrict__ input,        // [S, H_IN, W_IN, C_IN]
    const half* __restrict__ dvalues,      // [S, H_OUT, W_OUT, C_OUT]
    float* __restrict__ dweights,          // [K_TOTAL, C_OUT]
    float* __restrict__ dbiases,           // [C_OUT]
    const long long M_total                // S * H_OUT * W_OUT
) {
    // Implicit GEMM mapping: dW = X^T [K_TOTAL, M_total] * dY [M_total, C_OUT]
    //
    // X^T is produced by the *fragment layout*, not by a transposed gather:
    // As is staged [pixel][filter] (the orientation that makes the im2col
    // gather coalesced along C_IN) and read back with a column-major A
    // fragment, which reinterprets element (i,j) as As[j][i] == X^T. This
    // costs nothing -- no LDS transpose, no extra barrier.
    const int k_tile_start = blockIdx.x * BLOCK_TILE_M;
    const int n_tile_start = blockIdx.y * BLOCK_TILE_N;

    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;

    __shared__ half As[WMMA_K][A_LD];
    __shared__ half Bs[WMMA_K][BLOCK_TILE_N + B_PAD];
    __shared__ float s_bias[BLOCK_TILE_N];

    fragment<$matrix_a_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_a_col> a_frag;
    fragment<$matrix_b_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_b> b_frag;
    fragment<$accumulator_tag, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    float bias_acc[BIAS_ITEMS];
    #pragma unroll
    for (int i = 0; i < BIAS_ITEMS; ++i) bias_acc[i] = 0.0f;

    if (blockIdx.x == 0 && threadIdx.x < BLOCK_TILE_N) {
        s_bias[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // --- Hoisted A-gather addressing -------------------------------------
    // The filter index a_lane_k depends only on threadIdx.x and blockIdx.x
    // (THREADS is a multiple of BLOCK_TILE_M, so the strided loop below never
    // changes it), so the whole (fh, fw, c_in) decomposition and its input
    // row/col offsets are invariant across the entire reduction loop. The
    // original kernel recomputed them on every k-tile -- thousands of times.
    const int a_lane_k  = threadIdx.x % BLOCK_TILE_M;   // filter idx within tile
    const int a_lane_m0 = threadIdx.x / BLOCK_TILE_M;   // first pixel row for this lane
    const int a_rows_per_pass = THREADS / BLOCK_TILE_M;

    const int k_global_a = k_tile_start + a_lane_k;
    const bool a_k_valid = (k_global_a < K_TOTAL);
    const int c_in_a = a_k_valid ? (k_global_a % C_IN)        : 0;
    const int fw_a   = a_k_valid ? ((k_global_a / C_IN) % F_W) : 0;
    const int fh_a   = a_k_valid ? (k_global_a / C_IN_FW)      : 0;

    // Split-K reduction distribution across gridDim.z
    const int num_m_tiles = (int)((M_total + WMMA_K - 1) / WMMA_K);
    const int tiles_per_slice = (num_m_tiles + gridDim.z - 1) / gridDim.z;
    const int mt_start = blockIdx.z * tiles_per_slice;
    const int mt_end = (mt_start + tiles_per_slice < num_m_tiles) ? (mt_start + tiles_per_slice) : num_m_tiles;

    for (int mt = mt_start; mt < mt_end; ++mt) {

        // --- A tile: coalesced im2col gather, staged [pixel][filter] ---
        // Consecutive lanes walk consecutive filter indices, hence consecutive
        // c_in, hence contiguous `input` addresses. The LDS write is likewise
        // contiguous and bank-conflict free.
        #pragma unroll
        for (int local_m = a_lane_m0; local_m < WMMA_K; local_m += a_rows_per_pass) {
            const unsigned int m_global = (unsigned int)mt * WMMA_K + local_m;

            half val = __float2half(0.0f);
            if (a_k_valid && m_global < (unsigned int)M_total) {
                const unsigned int s     = m_global / HW_OUT;
                const unsigned int rem_m = m_global % HW_OUT;
                const int h_out = (int)(rem_m / W_OUT);
                const int w_out = (int)(rem_m % W_OUT);

                const int h_in = h_out * STR_H - PAD_H + fh_a;
                const int w_in = w_out * STR_W - PAD_W + fw_a;

                if (h_in >= 0 && h_in < H_IN && w_in >= 0 && w_in < W_IN) {
                    const long long in_idx = (((long long)s * H_IN + h_in) * W_IN + w_in) * C_IN + c_in_a;
                    val = input[in_idx];
                }
            }
            As[local_m][a_lane_k] = val;
        }

        // --- B tile: Load dY & register-accumulate bias ---
        if (blockIdx.x == 0) {
            int item = 0;
            for (int idx = threadIdx.x; idx < WMMA_K * BLOCK_TILE_N; idx += blockDim.x, ++item) {
                const int local_m = idx / BLOCK_TILE_N;
                const int local_n = idx % BLOCK_TILE_N;
                const unsigned int m_global = (unsigned int)mt * WMMA_K + local_m;
                const int n_global = n_tile_start + local_n;

                half val = __float2half(0.0f);
                if (m_global < (unsigned int)M_total && n_global < C_OUT) {
                    val = dvalues[(long long)m_global * C_OUT + n_global];
                    bias_acc[item] += __half2float(val);
                }
                Bs[local_m][local_n] = val;
            }
        } else {
            for (int idx = threadIdx.x; idx < WMMA_K * BLOCK_TILE_N; idx += blockDim.x) {
                const int local_m = idx / BLOCK_TILE_N;
                const int local_n = idx % BLOCK_TILE_N;
                const unsigned int m_global = (unsigned int)mt * WMMA_K + local_m;
                const int n_global = n_tile_start + local_n;

                half val = __float2half(0.0f);
                if (m_global < (unsigned int)M_total && n_global < C_OUT) {
                    val = dvalues[(long long)m_global * C_OUT + n_global];
                }
                Bs[local_m][local_n] = val;
            }
        }

        __syncthreads();

        if (warp_row < WARPS_M && warp_col < WARPS_N) {
            // col_major: element (i, j) == ptr[i + j * ldm] == As[pixel j][filter i]
            load_matrix_sync(a_frag, &As[0][warp_row * WMMA_M], A_LD);
            load_matrix_sync(b_frag, &Bs[0][warp_col * WMMA_N], BLOCK_TILE_N + B_PAD);
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }

    __shared__ float Cs[BLOCK_TILE_M][BLOCK_TILE_N];
    if (warp_row < WARPS_M && warp_col < WARPS_N) {
        store_matrix_sync(&Cs[warp_row * WMMA_M][warp_col * WMMA_N], acc_frag, BLOCK_TILE_N, $mem_layout);
    }
    __syncthreads();

    // --- Epilogue: dweights global store / accumulation ---
    for (int idx = threadIdx.x; idx < BLOCK_TILE_M * BLOCK_TILE_N; idx += blockDim.x) {
        const int local_k = idx / BLOCK_TILE_N;
        const int local_n = idx % BLOCK_TILE_N;
        const int k_global = k_tile_start + local_k;
        const int n_global = n_tile_start + local_n;

        if (k_global < K_TOTAL && n_global < C_OUT) {
            const float val = Cs[local_k][local_n];
            if (gridDim.z > 1) {
                atomicAdd(&dweights[k_global * C_OUT + n_global], val);
            } else {
                dweights[k_global * C_OUT + n_global] = val;
            }
        }
    }

    // --- Epilogue: Local bias flush to shared memory ---
    if (blockIdx.x == 0) {
        int item = 0;
        for (int idx = threadIdx.x; idx < WMMA_K * BLOCK_TILE_N; idx += blockDim.x, ++item) {
            const int local_n = idx % BLOCK_TILE_N;
            if (bias_acc[item] != 0.0f) {
                atomicAdd(&s_bias[local_n], bias_acc[item]);
            }
        }
    }
    __syncthreads();

    // --- Epilogue: Atomic global write for dbiases per slice ---
    if (blockIdx.x == 0 && threadIdx.x < BLOCK_TILE_N) {
        const int n_global = n_tile_start + threadIdx.x;
        if (n_global < C_OUT) {
            atomicAdd(&dbiases[n_global], s_bias[threadIdx.x]);
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
#define HW_OUT  (H_OUT * W_OUT)
#define C_IN_FW (C_IN * F_W)

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
    float* __restrict__ dinputs,           // [S, H_IN, W_IN, C_IN] - zero initialized
    const long long M_total
) {
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
            const unsigned int m_global = (unsigned int)(m_tile_start + local_m);
            const int c_global = ct * WMMA_K + local_c;

            half val = __float2half(0.0f);
            if (m_global < (unsigned int)M_total && c_global < C_OUT) {
                val = dvalues[(long long)m_global * C_OUT + c_global];
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
        const unsigned int m_global = (unsigned int)(m_tile_start + local_m);
        const int k_global = k_tile_start + local_k;

        if (m_global < (unsigned int)M_total && k_global < K_TOTAL) {
            const unsigned int s     = m_global / HW_OUT;
            const unsigned int rem_m = m_global % HW_OUT;
            const int h_out = (int)(rem_m / W_OUT);
            const int w_out = (int)(rem_m % W_OUT);

            const int c_in = k_global % C_IN;
            const int fw   = (k_global / C_IN) % F_W;
            const int fh   = k_global / C_IN_FW;

            const int h_in = h_out * STR_H - PAD_H + fh;
            const int w_in = w_out * STR_W - PAD_W + fw;

            if (h_in >= 0 && h_in < H_IN && w_in >= 0 && w_in < W_IN) {
                const long long in_idx = (((long long)s * H_IN + h_in) * W_IN + w_in) * C_IN + c_in;
                atomicAdd(&dinputs[in_idx], Cs[local_m][local_k]);
            }
        }
    }
}
''')

# --- Kernel Caching and Dispatch --------------------------------------------

_CONV_FORWARD_KERNEL_CACHE = {}
_CONV_BACKWARD_DWEIGHT_CACHE = {}
_CONV_BACKWARD_DINPUT_CACHE = {}


def get_compiled_forward_conv_kernel(shape_meta: ConvShapeMeta):
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


def get_compiled_backward_dweight_kernel(shape_meta: ConvShapeMeta):
    key = shape_meta.cache_key
    if key in _CONV_BACKWARD_DWEIGHT_CACHE:
        return _CONV_BACKWARD_DWEIGHT_CACHE[key]

    vendor = _CONV_HIP if shape_meta.is_hip else _CONV_CUDA
    block_tile_m, block_tile_n = _BLOCK_TILE_M, _BLOCK_TILE_N
    source = CNN_BACKWARD_DWEIGHT_TEMPLATE.substitute(
        c_in=shape_meta.C_in, c_out=shape_meta.C_out,
        f_h=shape_meta.fH, f_w=shape_meta.fW,
        str_h=shape_meta.sH, str_w=shape_meta.sW,
        pad_h=shape_meta.pad_h, pad_w=shape_meta.pad_w,
        h_in=shape_meta.H_in, w_in=shape_meta.W_in,
        h_out=shape_meta.H_out, w_out=shape_meta.W_out,
        wmma_dim=_WMMA_DIM,
        block_tile_m=block_tile_m, block_tile_n=block_tile_n,
        a_pad=_DWEIGHT_A_PAD, b_pad=_DWEIGHT_B_PAD,
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

    threads_per_block = (block_tile_m // _WMMA_DIM) * (block_tile_n // _WMMA_DIM) * vendor['warp_size']
    launch_meta = dict(
        block=(threads_per_block,),
        block_tile_m=block_tile_m,
        block_tile_n=block_tile_n,
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

# --- Launch planning --------------------------------------------------------
#
# Grid geometry, split-K and kernel argument order live here, not in callers.
# Layers, benchmarks and profilers all go through plan_*_launch, so a script
# cannot drift from what the layer actually runs -- an earlier profiler built
# the dweight grid by hand, omitted the split-K dimension, and made the kernel
# look ~10x slower than it is in production.

@dataclass(frozen=True)
class ConvLaunchPlan:
    """A compiled kernel bound to a concrete grid, block and M_total."""
    kernel: object
    grid: tuple
    block: tuple
    M_total: object
    shape_meta: ConvShapeMeta

    def __call__(self, *tensors):
        """Launch. Every conv kernel takes M_total as its trailing argument."""
        self.kernel(self.grid, self.block, (*tensors, self.M_total))

    @property
    def num_blocks(self):
        n = 1
        for d in self.grid:
            n *= d
        return n


def _m_total(shape_meta: ConvShapeMeta, S: int):
    return np.int64(S * shape_meta.H_out * shape_meta.W_out)


def plan_forward_launch(shape_meta: ConvShapeMeta, S: int):
    """conv_forward_wmma(input_fp16, weight_fp16, bias, output) -> plan."""
    compiled = get_compiled_forward_conv_kernel(shape_meta)
    if compiled is None:
        return None
    kernel, launch = compiled

    M_total = _m_total(shape_meta, S)
    grid = (
        int(-(-M_total // launch["block_tile_m"])),
        int(-(-shape_meta.C_out // launch["block_tile_n"])),
    )
    return ConvLaunchPlan(kernel, grid, launch["block"], M_total, shape_meta)


def plan_dweight_launch(shape_meta: ConvShapeMeta, S: int):
    """conv_backward_dweight_wmma(input_fp16, dvalues_fp16, dweights, dbiases).

    Note the 3D grid: gridDim.z carries the split-K slices that make this
    kernel fast. dweights/dbiases must be zeroed by the caller whenever
    gridDim.z > 1, since the epilogue accumulates with atomicAdd.
    """
    compiled = get_compiled_backward_dweight_kernel(shape_meta)
    if compiled is None:
        return None
    kernel, launch = compiled

    M_total = _m_total(shape_meta, S)
    grid_x = int(-(-shape_meta.K_total // launch["block_tile_m"]))
    grid_y = int(-(-shape_meta.C_out // launch["block_tile_n"]))
    num_m_tiles = int(-(-M_total // _WMMA_DIM))
    split_k = choose_dweight_split_k(num_m_tiles, grid_x, grid_y)

    return ConvLaunchPlan(
        kernel, (grid_x, grid_y, split_k), launch["block"], M_total, shape_meta
    )


def plan_dinput_launch(shape_meta: ConvShapeMeta, S: int):
    """conv_backward_dinput_wmma(dvalues_fp16, weight_fp16, dinputs).

    dinputs must be zeroed by the caller -- the epilogue scatter-adds.
    """
    compiled = get_compiled_backward_dinput_kernel(shape_meta)
    if compiled is None:
        return None
    kernel, launch = compiled

    M_total = _m_total(shape_meta, S)
    grid = (
        int(-(-M_total // launch["block_tile_m"])),
        int(-(-shape_meta.K_total // launch["block_tile_n"])),
    )
    return ConvLaunchPlan(kernel, grid, launch["block"], M_total, shape_meta)
