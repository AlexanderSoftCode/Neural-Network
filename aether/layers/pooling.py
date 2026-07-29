from functools import partial
import numpy as np
import aether.config as config
from aether.base import Layer
from aether.custom_kernels import pooling_kernel as gpu_pooling

class _PoolNd(Layer):
    def _get_padded_input(self, H_in, W_in):
        fH, fW = self.filter_size
        sH, sW = self.stride

        if self.padding == "valid":
            H_out = int(np.floor((H_in - fH) / sH + 1).item())
            W_out = int(np.floor((W_in - fW) / sW + 1).item())

            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        elif self.padding == "same":

            H_out = int(np.ceil(H_in / sH).item())
            W_out = int(np.ceil(W_in / sW).item())

            pad_h = max((H_out - 1) * sH + fH - H_in, 0)
            pad_w = max((W_out - 1) * sW + fW - W_in, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        else:
            raise ValueError(f"Expected padding == valid or same, recieved {self.padding} instead")
        return H_out, W_out, pad_top, pad_bottom, pad_left, pad_right

class MaxPool2d(_PoolNd):
    def __init__(self, filter_size = (2, 2), stride = (2,2),
                 padding = 'valid'):

        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.forward = self._forward_fallback
        self.backward = self._backward_fallback
    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to map low-level hardware paths."""
        if device == 'cupy' and gpu_pooling.is_gpu_max_pool2d_available():
            is_hip = config.HAS_CUPY and config.xp.cuda.runtime.is_hip
            self._variant = "hip" if is_hip else "cuda"

            # Block depth of 2 is recommended for HIP
            # Block depth of 4 is recommended for CUDA
            self._block_z = 2 if is_hip else 4

            self._kernel_train = gpu_pooling.get_max_pool2d_forward_kernel(
                self._variant, training=True
            )
            self._kernel_infer = gpu_pooling.get_max_pool2d_forward_kernel(
                self._variant, training=False
            )

            self._kernel_backward_nonoverlap = gpu_pooling.get_max_pool2d_backward_kernel(
                self._variant
            )
            self._launch_cache = {}

            self.forward = partial(self._forward_gpu, block_z=self._block_z)
            self.backward = partial(self._backward_gpu, block_z=self._block_z)
        else:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback

    def _get_launch_geometry(self, S, H_pad, W_pad, C, H_out, W_out, block_z):
        """A helper function to memoize grid/block/static argument per input shape"""

        key = (S, H_pad, W_pad, C, H_out, W_out, block_z)
        cached = self._launch_cache.get(key)
        if cached is not None:
            return cached

        fH, fW = self.filter_size
        sH, sW = self.stride

        block_x = min(32, C)
        block_y = 8
        block_dim = (block_x, block_y, block_z)
 
        grid_x = (C + block_x - 1) // block_x
        grid_y = (W_out + block_y - 1) // block_y
        grid_z = (H_out * S + block_z - 1) // block_z
        grid_dim = (grid_x, grid_y, grid_z)
 
        static_args = (
            np.int32(S), np.int32(H_pad), np.int32(W_pad), np.int32(C),
            np.int32(fH), np.int32(fW), np.int32(sH), np.int32(sW),
            np.int32(H_out), np.int32(W_out),
        )
 
        result = (block_dim, grid_dim, static_args)
        self._launch_cache[key] = result
        return result
    
    def _forward_gpu(self, inputs, training, block_z): 
        """Unified GPU forward path for both CUDA and ROCm/HIP."""
        xp = config.get_array_module(inputs)
        if inputs.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, recieved {inputs.ndim}D tensor instead")

        inputs = inputs.astype(xp.float32, copy=False)
        S, H_in, W_in, C = inputs.shape
        fH, fW = self.filter_size
        sH, sW = self.stride
        H_out, W_out, pad_top, pad_bottom, pad_left, pad_right = self._get_padded_input(H_in, W_in)

        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right

        if self.padding == "valid":
            inputs_padded = inputs
        elif self.padding == "same":
            inputs_padded = xp.pad(
                inputs,
                (
                    (0, 0),
                    (self.pad_top, self.pad_bottom),
                    (self.pad_left, self.pad_right),
                    (0, 0),
                ),
                mode='constant',
                constant_values=-np.inf
            )
        else: 
            raise ValueError(f"Expected padding == valid or same, recieved {self.padding} instead")

        inputs_padded = xp.ascontiguousarray(inputs_padded)
        _, H_pad, W_pad, _ = inputs_padded.shape

        # Save the two shapes for backward_gpu to un-pad dinputs
        self.inputs_shape = inputs.shape
        self.padded_shape = inputs_padded.shape

        self.output = xp.empty((S, H_out, W_out, C), dtype = inputs.dtype)
        block_dim, grid_dim, static_args = self._get_launch_geometry(
            S, H_pad, W_pad, C, H_out, W_out, block_z
        )

        if training:
            self.max_indices = xp.empty((S, H_out, W_out, C), dtype=xp.int32)
            kernel = self._kernel_train
            kernel_args = (inputs_padded, self.output, self.max_indices) + static_args
        else:
            self.max_indices = None
            kernel = self._kernel_infer
            kernel_args = (inputs_padded, self.output) + static_args

        kernel(grid_dim, block_dim, kernel_args)

        return self.output 

    def _backward_gpu(self, dvalues, block_z):
        
        if self.filter_size == self.stride:
            xp = config.get_array_module(dvalues)

            S, H_pad, W_pad, C = self.padded_shape
            _, H_out, W_out, _ = dvalues.shape
            _, H_in, W_in, _ = self.inputs_shape

            dvalues = xp.ascontiguousarray(dvalues.astype(xp.float32, copy=False))
            dinputs_padded = xp.zeros(self.padded_shape, dtype=dvalues.dtype)

            block_dim, grid_dim, static_args = self._get_launch_geometry(
                S, H_pad, W_pad, C, H_out, W_out, block_z
            )

            kernel = self._kernel_backward_nonoverlap
            kernel_args = (dvalues, self.max_indices, dinputs_padded) + static_args
            kernel(grid_dim, block_dim, kernel_args)

            self.dinputs = dinputs_padded[:,
                              self.pad_top : self.pad_top + H_in,
                              self.pad_left : self.pad_left + W_in,
                              :]
            return self.dinputs

        # Slower fallback path (overlapping windows)
        else:
            # Resue vectorized scatter-add logic via CuPy/xp.add.at
            return self._backward_fallback(dvalues)
    
    def _forward_fallback(self, inputs, training):
          
        xp = config.get_array_module(inputs)
        as_strided = config.get_stride_utility(xp)
        if inputs.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, recieved {inputs.ndim}D tensor instead")
        
        self.inputs_shape = inputs.shape
        S, H_in, W_in, C = inputs.shape
        fH, fW = self.filter_size
        sH, sW = self.stride
        # Unpack the returned tuple once cleanly
        H_out, W_out, pad_top, pad_bottom, pad_left, pad_right = self._get_padded_input(H_in, W_in)

        self.pad_top, self.pad_bottom = pad_top, pad_bottom
        self.pad_left, self.pad_right = pad_left, pad_right
        if self.padding == "valid":
            inputs_padded = inputs
        else:  # "same"
            inputs_padded = xp.pad(
                inputs,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant', constant_values=-np.inf
            )

        inputs_padded = xp.ascontiguousarray(inputs_padded)
        self.padded_shape = inputs_padded.shape
        _, H_pad, W_pad, _ = inputs_padded.shape

        patches = as_strided(
            inputs_padded,
            shape = (S, H_out, W_out, fH, fW, C), 
            strides = (
                inputs_padded.strides[0],      # Step between samples
                inputs_padded.strides[1] * sH, # Step between rows
                inputs_padded.strides[2] * sW, # Step between columns
                inputs_padded.strides[1],      # Move down 1 row inside patch
                inputs_padded.strides[2],      # Move right 1 col inside patch
                inputs_padded.strides[3],      # Step between each channel
            )
        )
        pooled = patches.max(axis = (3, 4)) 
        flat_in_window = patches.reshape(S, H_out, W_out, fH * fW, C).argmax(axis=3)
        win_rows, win_cols = xp.unravel_index(flat_in_window, (fH, fW))

        # Absolute row/col in the *padded* input
        h_out_idx = xp.arange(H_out).reshape(1, H_out, 1, 1)
        w_out_idx = xp.arange(W_out).reshape(1, 1, W_out, 1)
        s_idx = xp.arange(S).reshape(S, 1, 1, 1)
        c_idx = xp.arange(C).reshape(1, 1, 1, C)

        h_in = h_out_idx * sH + win_rows
        w_in = w_out_idx * sW + win_cols

        # Same flat-index convention as the GPU kernel: ((s*H_pad + h_in)*W_pad + w_in)*C + c
        self.max_indices = (((s_idx * H_pad + h_in) * W_pad + w_in) * C + c_idx).astype(xp.int32)

        self.output = pooled
        return self.output

    def _backward_fallback(self, dvalues):
        xp = config.get_array_module(dvalues)

        S, H_pad, W_pad, C = self.padded_shape
        _, H_in, W_in, _ = self.inputs_shape
        flat_dinputs = xp.zeros(S * H_pad * W_pad * C, dtype=dvalues.dtype)
        xp.add.at(flat_dinputs, self.max_indices.ravel(), dvalues.ravel())
        dinputs_padded = flat_dinputs.reshape(S, H_pad, W_pad, C)

        self.dinputs = dinputs_padded[:, self.pad_top:self.pad_top + H_in,
                                        self.pad_left:self.pad_left + W_in, :]
        return self.dinputs

    