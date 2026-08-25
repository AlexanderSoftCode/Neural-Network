import aether.config as config
import tests.base_case as base_case

from aether.layers.conv import Conv2d


class TestConvLayer(base_case.AetherBaseLayerTestCase):
    SEED = 42
    CONV_SHAPE = (2, 28, 28, 1)
    IN_CHANNELS=1
    OUT_CHANNELS=4
    FILTER_SIZE=(3,3)
    STRIDE=(1,1)
    PADDING="same"
    def setUp(self):
        super().setUp()
        self.as_strided = config.get_stride_utility(self.xp)
        
        self.layer = self.make_built_layer(
            Conv2d, 
            input_shape= self.CONV_SHAPE[1:],
            seed=self.SEED,
            in_channels=self.IN_CHANNELS, 
            out_channels=self.OUT_CHANNELS, 
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        #Dummy image, batch of 2, 28x28 pixels, 1 channel
        self.test_images = self.xp.random.randn(*self.CONV_SHAPE)

    def test_conv_forward_shape(self):
        """Verify output dimensions based on padding and stride"""
        output = self.layer.forward(self.test_images, training=True)
        # Given padding = 1, kernel = 3, 28x28 remainds 28x28 (same)
        self.assertEqual(output.shape, (2, 28, 28, 4))
    
    def test_parameters_initialization(self):
        """Compare filter weights and filter bias shapes"""
        self.assertEqual(self.layer.filter_weights.shape, (3, 3, 1, 4))
        self.assertEqual(self.layer.biases.shape, (4,))

    def test_backward_gradient_shapes(self):
        """Verify backprop gradients match weight shapes"""
        output = self.layer.forward(self.test_images, training=True)
        dvalues = self.xp.random.randn(*output.shape) #mock gradient
        self.layer.backward(dvalues)

        self.assertEqual(self.layer.dweights.shape, self.layer.filter_weights.shape)
        self.assertEqual(self.layer.dbiases.shape, self.layer.biases.shape)

    def test_conv_same_padding_stride(self):
        # 28x28 input, 3x3 filter, stride=(2,2), padding='same'
        layer = self.make_built_layer(
            Conv2d,
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS, 
            out_channels=self.OUT_CHANNELS, 
            filter_size=self.FILTER_SIZE,
            stride=(2,2),
            padding=self.PADDING
        )
        output = layer.forward(self.test_images, training=True)
        
        # Calculation: P=1 -> int((28 + 2 - 3)/2 + 1) = 14
        self.assertEqual(output.shape, (2, 14, 14, 4))
        
    def test_conv_valid_padding_stride(self):
        # 28x28 input, 3x3 filter, stride = (2,2), valid padding
        layer = self.make_built_layer(
            Conv2d, 
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS, 
            out_channels=self.OUT_CHANNELS, 
            filter_size=self.FILTER_SIZE,
            stride=(2,2),
            padding="valid"
        )
        output = layer.forward(self.test_images, training=True)
        # (28 - 3) / 2 + 1 = 13.5 -> floor to 13
        self.assertEqual(output.shape, (2, 13, 13, 4))

    def test_conv_zero_input(self):
        "Given an input of all zeros, show an output of all zeros (plus biases)"

        zero_input = self.xp.zeros((1, 28, 28, 1))
        
        # Expected: Every pixel in the output should equal the bias for that filter
        # We reshape biases to (1, 1, 1, num_filters) to match output shape
        
        output = self.layer.forward(zero_input, training = False)
        expected_output = self.xp.broadcast_to(self.layer.biases, output.shape)
        
        self.xp.testing.assert_array_almost_equal(output, expected_output)

    def test_conv_stride_length(self):

        layer = self.make_built_layer(
            Conv2d,
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=(2,1),
            padding=self.PADDING
        )
        output = layer.forward(self.test_images, training = False)
        
        self.assertEqual(output.shape, (2, 14, 28, 4))

    def test_conv_stride_width(self):
        layer = self.make_built_layer(
            Conv2d,
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=(1,2),
            padding=self.PADDING
        )
        output = layer.forward(self.test_images, training=False)
        self.assertEqual(output.shape, (2, 28, 14, 4))

    def test_conv_multichannel_input(self):
        """Verify forward pass with multi-channel input (e.g., RGB with C_in=3)"""
        layer = self.make_built_layer(
            Conv2d,
            input_shape = (28, 28, 3),
            in_channels=3,
            out_channels=8,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        rgb_images = self.xp.random.randn(2, 28, 28, 3)
        output = layer.forward(rgb_images, training=False)
        self.assertEqual(output.shape, (2, 28, 28, 8))

    def test_conv_variable_batch_sizes(self):
        """Verify shape cache handles dynamic batch sizes without re-compilation or shape errors"""
        inputs_b4 = self.xp.random.randn(4, 16, 16, self.IN_CHANNELS)
        out_b4 = self.layer.forward(inputs_b4, training=False)
        self.assertEqual(out_b4.shape, (4, 16, 16, self.OUT_CHANNELS))

        inputs_b1 = self.xp.random.randn(1, 16, 16, self.IN_CHANNELS)
        out_b1 = self.layer.forward(inputs_b1, training=False)
        self.assertEqual(out_b1.shape, (1, 16, 16, self.OUT_CHANNELS))

    def test_conv_non_square_filter(self):
        """Verify rectangular filter geometries such as (5, 3)"""
        layer = self.make_built_layer(
            Conv2d,
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=(5, 3),
            stride=(1, 1),
            padding="same"
        )
        output = layer.forward(self.test_images, training=False)
        self.assertEqual(output.shape, (2, 28, 28, self.OUT_CHANNELS))

    # ---- Conv.build() Seed Tests ----------

    def test_same_seed_produces_identical_weights(self):
        """Two independently instantiated layers built with the exact same seed
        must initialize with identical weight and bias values."""
        layer_1 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        layer_2 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )

        seed_val = 123
        layer_1.build(input_shape = self.CONV_SHAPE[1:], seed=seed_val)
        layer_2.build(input_shape = self.CONV_SHAPE[1:], seed=seed_val)

        # Weights and biases must match exactly bit-for-bit on the active backend
        self.xp.testing.assert_array_equal(
            layer_1.weights, layer_2.weights,
            err_msg="Layers built with identical seeds produced diverging weights."
        )
        self.xp.testing.assert_array_equal(
            layer_1.biases, layer_2.biases,
            err_msg="Layers built with identical seeds produced diverging biases."
        )

    def test_different_seeds_produce_divergent_weights(self):
        """Two layers built with different seeds must produce different parameter initializations."""
        layer_1 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        layer_2 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )

        layer_1.build(input_shape=self.CONV_SHAPE[1:], seed=123)
        layer_2.build(input_shape=self.CONV_SHAPE[1:], seed=456)

        # Weights should not be identical
        self.assertFalse(
            self.xp.array_equal(layer_1.weights, layer_2.weights),
            msg="Layers built with different seeds produced identical weights."
        )

    def test_build_seed_does_not_corrupt_global_rng_stream(self):
        """Calling build(seed=...) must draw from a local PRNG and avoid resetting
        or corrupting the backend's ambient global random state."""
        xp = self.xp

        # Seed global backend state and sample a reference number
        xp.random.seed(999)
        val_before = float(xp.random.randn(1)[0])

        # Reset global state to 999, build an intermediate layer with an isolated seed,
        # and verify the next global draw is unaffected
        xp.random.seed(999)
        throwaway_layer = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        throwaway_layer.build(input_shape = self.CONV_SHAPE[1:], seed=123)

        val_after = float(xp.random.randn(1)[0])

        self.assertAlmostEqual(
            val_before, val_after, places=6,
            msg="layer.build(seed=...) modified or reset the global backend random state."
        )

    def test_build_with_none_seed_uses_global_stream(self):
        """Calling build(seed=None) should sequentially consume random numbers
        from the global stream rather than throwing an error or using a static seed."""
        xp = self.xp

        xp.random.seed(777)
        layer_1 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        layer_1.build(input_shape = self.CONV_SHAPE[1:], seed=None)

        xp.random.seed(777)
        layer_2 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        layer_2.build(input_shape = self.CONV_SHAPE[1:], seed=None)

        self.xp.testing.assert_array_equal(
            layer_1.weights,
            layer_2.weights,
            err_msg="build(seed=None) failed to follow global backend seeding."
        )

    # ---- Gradient Check -----------------
    def test_conv_numerical_gradient_check(self):
        """
        Finite difference check: verify analytical dweights matches numerical approximation
        for the Conv. Uses a small input/filter to keep the double loop feasible.
        f'(x) ≈ (f(x + h) - f(x - h)) / 2h
        """
        epsilon = 1e-2
        layer = self.make_built_layer(
            Conv2d,
            input_shape = (8, 8, 1),
            in_channels=self.IN_CHANNELS,
            out_channels=2,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
    
        fixed_input = self.xp.random.randn(2, 8, 8, 1)  # batch=2, 8x8, 1 channel
        dvalues = self.xp.ones((2, 8, 8, 2))             # ones so sum(output) is the scalar loss
    
        # Analytical gradient
        layer.forward(fixed_input, training=False)
        layer.backward(dvalues)
        analytical_dweights = layer.dweights.copy()
    
        # Numerical gradient — iterate over every element of filter_weights (3x3x1x2 = 18)
        numerical_dweights = self.xp.zeros_like(layer.filter_weights)
    
        for fh in range(layer.filter_weights.shape[0]):
            for fw in range(layer.filter_weights.shape[1]):
                for c in range(layer.filter_weights.shape[2]):
                    for f in range(layer.filter_weights.shape[3]):
    
                        # f(x + h) -- mutate the buffer in place, then invalidate,
                        # exactly like a fused optimizer kernel writing into
                        # layer.weights and calling invalidate_shadow_caches().
                        layer.filter_weights[fh, fw, c, f] += epsilon
                        layer.invalidate_shadow_caches()
                        layer.forward(fixed_input, training=False)
                        loss_plus = self.xp.sum(layer.output)
    
                        # f(x - h)
                        layer.filter_weights[fh, fw, c, f] -= 2 * epsilon
                        layer.invalidate_shadow_caches()
                        layer.forward(fixed_input, training=False)
                        loss_minus = self.xp.sum(layer.output)
    
                        # Restore -- also needs invalidating, so the layer is
                        # left in a correct state for the next element's pass
                        # (and for anything run after this loop).
                        layer.filter_weights[fh, fw, c, f] += epsilon
                        layer.invalidate_shadow_caches()
    
                        numerical_dweights[fh, fw, c, f] = (loss_plus - loss_minus) / (2 * epsilon)
    
        self.xp.testing.assert_allclose(
            analytical_dweights, 
            numerical_dweights, 
            rtol=3e-2,   # 3% relative tolerance
            atol=2e-2    # 0.02 absolute tolerance
        )
    # --- regression tests ---------------------------------------------------

    def test_fp16_cache_invalidation_on_weight_update(self):
        """Ensure updating layer weights invalidates the FP16 shadow cache"""
        if self.backend_name == 'cupy':
            self.layer._compile_for_device('cupy')
            _ = self.layer.forward(self.test_images, training=False)
            
            # If GPU matrix-core path was taken, cache should now be valid
            if self.layer._fp16_weight_cache is not None:
                self.assertTrue(self.layer._fp16_weight_valid)

                # Update weights via property setter
                new_weights = self.xp.random.randn(*self.layer.filter_weights.shape).astype(self.xp.float32)
                self.layer.weights = new_weights
                self.assertFalse(self.layer._fp16_weight_valid)

    def test_forward_gpu_fallback_numeric_parity(self):
        """Verify numerical equivalence between _forward_gpu and _forward_fallback (CuPy backend only)"""
        if self.backend_name == 'cupy':
            from aether.custom_kernels import conv_kernel
            if conv_kernel.get_is_conv_gpu_available():
                fixed_input = self.xp.random.randn(2, 14, 14, self.IN_CHANNELS).astype(self.xp.float32)

                # Execute GPU matrix-core path
                gpu_out = self.layer._forward_gpu(fixed_input, training=False)

                # Execute CPU/CuPy fallback path
                fallback_out = self.layer._forward_fallback(fixed_input, training=False)

                # FP16 Tensor Cores introduce minor precision differences vs FP32 einsum
                self.xp.testing.assert_allclose(gpu_out, fallback_out, rtol=1e-2, atol=1e-2)

    GEMM_REL_TOL = 5e-3

    def _assert_gemm_close(self, actual, desired, name, tol=None):
        """Norm-relative comparison: max|actual - desired| / max|desired|."""
        tol = self.GEMM_REL_TOL if tol is None else tol
        scale = float(self.xp.max(self.xp.abs(desired)))
        if scale == 0.0:
            scale = 1.0
        rel = float(self.xp.max(self.xp.abs(actual - desired))) / scale
        self.assertLessEqual(
            rel, tol,
            f"{name}: relative error {rel:.3e} exceeds {tol:.1e} "
            f"(max|ref|={scale:.3e})",
        )

    def _run_dweight_kernel_parity(self, S, spatial, in_channels, out_channels,
                                   stride=None, padding=None):
        """Drive the fused dweight kernel once and check both of its outputs.

        conv_backward_dweight_wmma emits dweights AND dbiases from a single
        launch -- the bias reduction rides along on the dY tile the GEMM has
        already staged. They are therefore checked together, off one launch,
        rather than as two independent tests.
        """
        if self.backend_name != 'cupy':
            self.skipTest("CuPy backend required for GPU kernel tests")
        from aether.custom_kernels import conv_kernel
        if not conv_kernel.get_is_conv_gpu_available():
            self.skipTest("Matrix-core GPU kernels not available on this device")

        stride = stride or self.STRIDE
        padding = padding or self.PADDING
        H, W = spatial

        layer = self.make_built_layer(
            Conv2d,
            input_shape=(H, W, in_channels),
            in_channels=in_channels,
            out_channels=out_channels,
            filter_size=self.FILTER_SIZE,
            stride=stride,
            padding=padding,
        )

        fixed_input = self.xp.random.randn(S, H, W, in_channels).astype(self.xp.float32)
        meta = layer._get_shape_meta((H, W, in_channels))
        dvalues = self.xp.random.randn(
            S, meta.H_out, meta.W_out, out_channels
        ).astype(self.xp.float32)

        layer._forward_fallback(fixed_input, training=False)
        layer._backward_fallback(dvalues)
        fallback_dweights = layer.dweights.copy()
        fallback_dbiases = layer.dbiases.copy()

        layer._forward_gpu(fixed_input, training=False)
        layer._backward_gpu(dvalues)
        gpu_dweights = layer.dweights.copy()
        gpu_dbiases = layer.dbiases.copy()

        self.assertEqual(gpu_dweights.shape, fallback_dweights.shape)
        self.assertEqual(gpu_dbiases.shape, fallback_dbiases.shape)

        self._assert_gemm_close(gpu_dweights, fallback_dweights, "dweights")
        self._assert_gemm_close(gpu_dbiases, fallback_dbiases, "dbiases")

        dvalues_fp16 = dvalues.astype(self.xp.float16).astype(self.xp.float32)
        ref_dbiases = self.xp.sum(dvalues_fp16, axis=(0, 1, 2))
        self.xp.testing.assert_allclose(gpu_dbiases, ref_dbiases, rtol=1e-5, atol=1e-5)

    def test_conv_gpu_backward_dweight_kernel_parity(self):
        """Fused dweight kernel: dweights and dbiases against the fallback."""
        self._run_dweight_kernel_parity(
            S=2, spatial=(8, 8), in_channels=self.IN_CHANNELS, out_channels=2
        )

    def test_conv_gpu_backward_dweight_kernel_parity_multichannel(self):
        """Same, on a shape whose K_TOTAL and C_OUT span several block tiles.

        The 8x8x1 -> 2 case above fits inside one 64x64 output tile, so it
        never exercises the grid.x/grid.y tiling or the ragged tail masking.
        """
        self._run_dweight_kernel_parity(
            S=4, spatial=(16, 16), in_channels=8, out_channels=48
        )

    def test_conv_gpu_backward_dweight_kernel_parity_valid_strided(self):
        """Same, with valid padding and a stride > 1."""
        self._run_dweight_kernel_parity(
            S=2, spatial=(16, 16), in_channels=4, out_channels=8,
            stride=(2, 2), padding="valid",
        )

    def test_conv_gpu_backward_dinput_parity(self):
        """Verify numerical equivalence for dinputs between _backward_gpu and _backward_fallback"""
        if self.backend_name != 'cupy':
            self.skipTest("CuPy backend required for GPU kernel tests")
        from aether.custom_kernels import conv_kernel
        if not conv_kernel.get_is_conv_gpu_available():
            self.skipTest("Matrix-core GPU kernels not available on this device")

        layer = self.make_built_layer(
            Conv2d,
            input_shape = (8, 8, self.IN_CHANNELS),
            in_channels=self.IN_CHANNELS,
            out_channels=2,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )

        fixed_input = self.xp.random.randn(2, 8, 8, self.IN_CHANNELS).astype(self.xp.float32)
        dvalues = self.xp.random.randn(2, 8, 8, 2).astype(self.xp.float32)

        # 1. Run Fallback End-to-End
        layer._forward_fallback(fixed_input, training=False)
        layer._backward_fallback(dvalues)
        fallback_dinputs = layer.dinputs.copy()

        # 2. Run GPU End-to-End
        layer._forward_gpu(fixed_input, training=False)
        layer._backward_gpu(dvalues)
        gpu_dinputs = layer.dinputs.copy()

        # WMMA FP16 + atomicAdd vs FP32 tensordot tolerance comparison
        self.xp.testing.assert_allclose(gpu_dinputs, fallback_dinputs, rtol=1e-2, atol=1e-2)


base_case.register_test_suites(globals(), TestConvLayer)