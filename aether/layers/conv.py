import numpy as np
import aether.config as config
from aether.base import Layer
from aether.custom_kernels import conv_kernel
class Conv(Layer):
    def __init__(self, in_channels, out_channels = 1, filter_size = (3, 3), stride = (1, 1), padding = "same"):

        # input_shape has form (batch_size, height, width, channels)
        self.C_in = in_channels
        self.C_out = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding 

        self.weight_regularizer_l1 = 0
        self.weight_regularizer_l2 = 0
        self.bias_regularizer_l1 = 0
        self.bias_regularizer_l2 = 0

        self.forward = self._forward_fallback
        self.backward = self._backward_fallback
        self._fp16_weight_cache = None
        self._fp16_weight_valid = False 
        self._launch_cache = {}

        self.weights = None
        self.biases = None

    def build(self):
        """
        Called once by Model.finalize(). config.xp is guaranteed to be
        correctly set if the user called model.to() beforehand.
        """
        xp = config.xp
        # We'll handle two scenarios, the first, where we pass in a (n, n, 1) or grayscale image, and a second
        # where we'll handle a (n, n, 3) or RGB image. 
        self.biases = xp.zeros(self.C_out, dtype = xp.float32)
        n = self.filter_size[0] * self.filter_size[1] * self.C_in
        std = xp.sqrt(xp.float32(2.0 / n))
        
        # We can now do He initaliztion, we'll sample values from a standard distribution N (0, 1) and multiply it by our
        # std value to get N(0, std) 
        self.filter_weights = (xp.random.randn(
            self.filter_size[0],         # Filter height fH
            self.filter_size[1],         # Filter width fW
            self.C_in,              # Input channels C_in 
            self.C_out              # Output channels C_out
        ).astype(xp.float32)* std)

        self.weights = self.filter_weights

    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to bind the matrix-core path when available."""
        if device == 'cupy' and conv_kernel.get_is_conv_gpu_available():
            self.forward = self._forward_gpu
            self.backward = self._backward_gpu
        else:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback
        # Device may have underlying weight buffers 0 force a shadow refresh
        self._fp16_weight_valid = False 

    def _refresh_fp16_weights(self, xp):
        """
        Shadow fp16 cast of filter weights, update optimizer kernels
        and classes to cast these back to f32, and back to f16
        """

        fH, fW, C_in, C_out = self.filter_weights.shape
        if self._fp16_weight_cache is None:
            self._fp16_weight_cache = xp.empty((fH * fW * C_in, C_out), dtype=xp.float16)
        if not self._fp16_weight_valid:
            self._fp16_weight_cache[...] = self.filter_weights.reshape(fH * fW * C_in, C_out).astype(xp.float16)
            self._fp16_weight_valid = True
    
    def invalidate_shadow_caches(self):
        """
        Called by an optimizer after it writes into self.weights'
        underlying buffer in place (e.g. a fused GPU kernel that takes
        the array as an output pointer) -- that kind of update never
        goes through the filter_weights/weights property setter, so
        _fp16_weight_valid would otherwise stay True and _forward_gpu
        would keep matrix-multiplying against a stale fp16 snapshot.
        Forces a rebuild on the next call that needs it.
        """
        self._fp16_weight_valid = False

    def _get_shape_meta(self, spatial_shape):
        """Retrieves cached launch parameters using spatial/channel dimensions
        AND the current stride/padding config (excluding batch S)."""
        fH, fW = self.filter_size
        sH, sW = self.stride

        pad_h = (fH - 1) // 2 if self.padding == "same" else 0
        pad_w = (fW - 1) // 2 if self.padding == "same" else 0

        H_in, W_in, C_in = spatial_shape
        cache_key = (H_in, W_in, C_in, sH, sW, pad_h, pad_w)

        cached = self._launch_cache.get(cache_key)
        if cached is not None:
            return cached

        H_out = int((H_in + 2 * pad_h - fH) / sH + 1)
        W_out = int((W_in + 2 * pad_w - fW) / sW + 1)

        shape_meta = conv_kernel.ConvShapeMeta(
            H_in=H_in, W_in=W_in, C_in=C_in, C_out=self.C_out,
            fH=fH, fW=fW, sH=sH, sW=sW, pad_h=pad_h, pad_w=pad_w,
            H_out=H_out, W_out=W_out, is_hip=conv_kernel._is_hip_backend(),
        )

        compiled = conv_kernel.get_compiled_forward_conv_kernel(shape_meta)
        if compiled is None:
            return None

        kernel, launch_meta = compiled

        meta = {
            "kernel": kernel,
            "block_dim": launch_meta["block"],
            "block_tile_m": launch_meta["block_tile_m"],
            "block_tile_n": launch_meta["block_tile_n"],
            "H_out": H_out,
            "W_out": W_out,
        }

        self._launch_cache[cache_key] = meta
        return meta

    def _forward_gpu(self, inputs, training):
        xp = config.get_array_module(inputs)
        S, H_in, W_in, C_in = inputs.shape
        meta = self._get_shape_meta((H_in, W_in, C_in))

        # Fall back if JIT failed or shape isn't supported by Matrix Cores
        if meta is None:
            self.forward = self._forward_fallback
            return self._forward_fallback(inputs, training)

        inputs_fp16 = inputs.astype(xp.float16, copy=False)

        if not self._fp16_weight_valid:
            self._refresh_fp16_weights(xp)

        H_out, W_out = meta["H_out"], meta["W_out"]
        M_total = np.int64(S * H_out * W_out)

        grid_dim = (
            int(-(-M_total // meta["block_tile_m"])),
            int(-(-self.C_out // meta["block_tile_n"])),
        )

        self.output = xp.empty((S, H_out, W_out, self.C_out), dtype=xp.float32)

        kernel_args = (
            inputs_fp16,
            self._fp16_weight_cache,
            self.biases,
            self.output,
            M_total,
        )

        meta["kernel"](grid_dim, meta["block_dim"], kernel_args)
        self.inputs = inputs
        return self.output

    def _backward_gpu(self, dvalues):
        xp = config.get_array_module(dvalues)
        S, H_in, W_in, C_in = self.inputs.shape
        fH, fW = self.filter_size
        sH, sW = self.stride

        pad_h = (fH - 1) // 2 if self.padding == "same" else 0
        pad_w = (fW - 1) // 2 if self.padding == "same" else 0

        H_out = int((H_in + 2 * pad_h - fH) / sH + 1)
        W_out = int((W_in + 2 * pad_w - fW) / sW + 1)

        # Build shape metadata signature
        shape_meta = conv_kernel.ConvShapeMeta(
            H_in=H_in, W_in=W_in, C_in=C_in, C_out=self.C_out,
            fH=fH, fW=fW, sH=sH, sW=sW, pad_h=pad_h, pad_w=pad_w,
            H_out=H_out, W_out=W_out, is_hip=conv_kernel._is_hip_backend(),
        )

        # Retrieve compiled kernels
        dw_compiled = conv_kernel.get_compiled_backward_dweight_kernel(shape_meta)
        dx_compiled = conv_kernel.get_compiled_backward_dinput_kernel(shape_meta)
        db_compiled = conv_kernel.get_compiled_backward_dbias_kernel(shape_meta)

        # Fall back if any GPU compile fails
        if dw_compiled is None or dx_compiled is None or db_compiled is None:
            raise RuntimeError(
                f"Failed to compile GPU backward kernels for Conv shape "
                f"(H_in={H_in}, W_in={W_in}, C_in={C_in}, C_out={self.C_out}). "
                f"Cannot fall back to CPU because forward pass executed on GPU."
            )

        M_total = np.int64(S * H_out * W_out)
        K_total = fH * fW * C_in

        # Prepare FP16 inputs for matrix-core WMMA operations
        dvalues_fp32 = dvalues.astype(xp.float32, copy=False)
        dvalues_bounded = xp.clip(dvalues_fp32, -65504.0, 65504.0)
        dvalues_fp16 = dvalues_bounded.astype(xp.float16, copy=False)

        inputs_fp16 = self.inputs.astype(xp.float16, copy=False)

        if not self._fp16_weight_valid:
            self._refresh_fp16_weights(xp)

        # --- 1. Bias Gradients (dbias) ---
        db_kernel, db_launch = db_compiled
        self.dbiases = xp.empty(self.C_out, dtype=xp.float32)
        grid_db = (int(-(-self.C_out // db_launch["block"][0])),)
        db_kernel(grid_db, db_launch["block"], (dvalues_fp32, self.dbiases, M_total, np.int32(self.C_out)))

        # --- 2. Weight Gradients (dweight) ---
        dw_kernel, dw_launch = dw_compiled
        self.dweights = xp.zeros((fH, fW, C_in, self.C_out), dtype=xp.float32)
        grid_dw = (
            int(-(-K_total // dw_launch["block_tile_m"])),
            int(-(-self.C_out // dw_launch["block_tile_n"])),
        )
        dw_kernel(grid_dw, dw_launch["block"], (inputs_fp16, dvalues_fp16, self.dweights, M_total))

        # --- 3. Input Gradients (dinput) ---
        dx_kernel, dx_launch = dx_compiled
        # Must be initialized to 0 because conv_backward_dinput_wmma uses atomicAdd
        self.dinputs = xp.zeros_like(self.inputs, dtype=xp.float32)
        grid_dx = (
            int(-(-M_total // dx_launch["block_tile_m"])),
            int(-(-K_total // dx_launch["block_tile_n"])),
        )
        dx_kernel(grid_dx, dx_launch["block"], (dvalues_fp16, self._fp16_weight_cache, self.dinputs, M_total))

        return self.dinputs
    
    def _forward_fallback(self, inputs, training):
        
        xp = config.get_array_module(inputs)
        as_strided = config.get_stride_utility(xp)
        fH, fW = self.filter_size
        sH, sW = self.stride
        S, H_in, W_in, D_in = inputs.shape

        pad_h = (fH - 1) // 2 if self.padding == "same" else 0
        pad_w = (fW - 1) // 2 if self.padding == "same" else 0

        # Retrieve spatial geometry - keyed on stride/pad too, since both
        # are mutable instance attrs and this cache is per-instance.
        spatial_key = (H_in, W_in, D_in, sH, sW, pad_h, pad_w, "fallback")
        meta = self._launch_cache.get(spatial_key)

        if meta is None:
            H_out = int((H_in + 2 * pad_h - fH) / sH + 1)
            W_out = int((W_in + 2 * pad_w - fW) / sW + 1)

            meta = {
                "pad_h": pad_h,
                "pad_w": pad_w,
                "H_out": H_out,
                "W_out": W_out,
            }
            self._launch_cache[spatial_key] = meta

        pad_h, pad_w = meta["pad_h"], meta["pad_w"]
        H_out, W_out = meta["H_out"], meta["W_out"] 
        # (0, 0) -> don't touch the number of samples in the batch
        # (P, P) -> pad top and bottom pixels by P pixels (axis 1)
        # (P, P) -> pad left and right pixels by P pixels (axis 2)
        # (0, 0) -> don't pad depth. 
        # contstant -> add constant_values for the padded values
        padded_inputs = xp.pad(array = inputs, 
                            pad_width = ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                            mode = 'constant',
                            constant_values = 0).astype(xp.float32, copy = False)

        self.output = xp.zeros((S, H_out, W_out, self.C_out), dtype = xp.float32)
        self.patches_shape=(S, H_out, W_out, fH, fW, D_in)
        self.patches_strides = (
                padded_inputs.strides[0],       # step between samples
                padded_inputs.strides[1] * sH,  # step down a row
                padded_inputs.strides[2] * sW,  # step across a column
                padded_inputs.strides[1],       # move down 1 row inside patch
                padded_inputs.strides[2],       # move right 1 col inside patch
                padded_inputs.strides[3],       # step across channels
        )

        patches = as_strided(
            padded_inputs,
            shape=self.patches_shape,
            strides=self.patches_strides)

        # Keep the samples, H_out, W_out, and C_out. 
        # But, iterate over the patch(x, y) with channels c, and with the number of filters d
        self.output = xp.einsum('shwxyc,xycd->shwd', patches, self.filter_weights, optimize=True)
        self.output += self.biases.reshape((1, 1, 1, self.C_out)) 

        self.inputs = inputs
        self.padded_inputs = padded_inputs
        return self.output

    
    def _backward_fallback(self, dvalues):

        xp = config.get_array_module(dvalues)
        as_strided = config.get_stride_utility(xp)

        S, H_out, W_out, C_out = dvalues.shape
        fH, fW, C_in, C_out = self.filter_weights.shape
        sH, sW = self.stride
        _, H_in, W_in, _ = self.inputs.shape 

        patches = as_strided(
            self.padded_inputs,
            shape=self.patches_shape,
            strides=self.patches_strides
        )
        # Now we need to account for dbiases and dweights
        self.dbiases = xp.sum(dvalues, axis = (0, 1, 2))
        self.dweights = xp.tensordot(patches, dvalues, axes = ([0, 1, 2], [0, 1, 2]))

        dilated_H = (H_out - 1) * sH + 1
        dilated_W = (W_out - 1) * sW + 1 
        
        dvalues_dilated = xp.zeros(shape= (S, dilated_H, dilated_W, C_out), dtype= dvalues.dtype)
        # Inject values using step slices
        dvalues_dilated[:, ::sH, ::sW, :] = dvalues
        
        pad_h = (fH - 1) // 2 if self.padding == "same" else 0
        pad_w = (fW - 1) // 2 if self.padding == "same" else 0

        if self.padding == "same": 
            backward_pad_h = (fH - 1) - pad_h
            backward_pad_w = (fW - 1) - pad_w
        if self.padding == "valid": 
            backward_pad_h = (fH - 1)
            backward_pad_w = (fW - 1)
        
        dvalues_padded = xp.pad(dvalues_dilated, pad_width= (
            (0, 0), (backward_pad_h, backward_pad_h), (backward_pad_w, backward_pad_w), (0, 0))
            )

        # flip the values in the fH and fW dimensions, leave C_in and C_out dimensions alone
        flipped_weights = self.filter_weights[::-1, ::-1, :, :]
        dvalues_patches = as_strided(dvalues_padded, 
                            shape = (S, H_in, W_in, fH, fW, C_out),
                            strides=(
                                dvalues_padded.strides[0],  # Batch step
                                dvalues_padded.strides[1],  # Window grid row step (backward stride = 1)
                                dvalues_padded.strides[2],  # Window grid col step (backward stride = 1)
                                dvalues_padded.strides[1],  # Internel window pixel row step
                                dvalues_padded.strides[2],  # Internel window pixel col step
                                dvalues_padded.strides[3]   # Output channel step
                            ))
    
        self.dinputs = xp.tensordot(
            dvalues_patches, 
            flipped_weights, 
            axes=([3, 4, 5], [0, 1, 3]) # Match fH, fW, C_out
        )

        return self.dinputs
    @property
    def filter_weights(self):
        return self._filter_weights

    @filter_weights.setter
    def filter_weights(self, value):
        # Automatically reshape 2D weight matrices to 4D filter tensors
        if value is not None and value.ndim == 2:
            fH, fW = self.filter_size
            value = value.reshape(fH, fW, self.C_in, self.C_out)
        self._filter_weights = value
        self._fp16_weight_valid = False

    @property
    def weights(self):
        return self.filter_weights

    @weights.setter
    def weights(self, value):
        self.filter_weights = value