import cupy as cp
from cupy.lib.stride_tricks import as_strided
class Conv_Layer:
    def __init__(self, input_shape, num_filters = 1, filter_size = (3, 3), strides = (1, 1), padding = "same"):

        #input_shape has form (batch_size, height, width, channels)
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding 
        self.biases = cp.zeros(self.num_filters, dtype = cp.float32) * 0.01
        self.weight_regularizer_l1 = 0
        self.weight_regularizer_l2 = 0
        self.bias_regularizer_l1 = 0
        self.bias_regularizer_l2 = 0

        #We'll handle two scenarios, the first, where we pass in a (n, n, 1) or grayscale image, and a second
        #where we'll handle a (n, n, 3) or RGB image. 
        input_depth = input_shape[-1]
        n = self.filter_size[0] * self.filter_size[1] * input_depth
        std = cp.sqrt(cp.float32(2.0 / n))
        
        #We can now do He initaliztion, we'll sample values from a standard distribution N (0, 1) and multiply it by our
        #std value to get N(0, std) 

        self.filter_weights = (cp.random.randn(
            filter_size[0],         #height
            filter_size[1],         #width
            input_depth,            #depth 
            num_filters             #number of filters
        ).astype(cp.float32)* std)

        self.weights = self.filter_weights

    def forward(self, inputs, training):
        #Extract Input dimensions

        fH, fW = self.filter_size
        sH, sW = self.strides
        S, H_in, W_in, D_in = inputs.shape
        
        #Creating padding depending on padding = same, or padding = valid
        if self.padding == "same":
            self.forward_pad_h = (self.filter_size[0] - 1) // 2
            self.forward_pad_w = (self.filter_size[1] - 1) // 2
        else:            
            self.forward_pad_h = 0
            self.forward_pad_w = 0

        #We need integer output dimensions, so cast equations to int
        H_out = int((H_in + 2 * self.forward_pad_h - self.filter_size[0]) / self.strides[0] + 1)
        W_out = int((W_in + 2 * self.forward_pad_w - self.filter_size[1]) / self.strides[1] + 1)
        
        #(0, 0) -> don't touch the number of samples in the batch
        #(P, P) -> pad top and bottom pixels by P pixels (axis 1)
        #(P, P) -> pad left and right pixels by P pixels (axis 2)
        #(0, 0) -> don't pad depth. 
        #contstant -> add constant_values for the padded values
        padded_inputs = cp.pad(array = inputs, 
                            pad_width = ((0, 0), (self.forward_pad_h, self.forward_pad_h),
                                          (self.forward_pad_w, self.forward_pad_w), (0, 0)),
                            mode = 'constant',
                            constant_values = 0).astype(cp.float32, copy = False)

        #Create an output tensor of size (batch_size, H_out, W_out, C_out)
        self.output = cp.zeros((S, H_out, W_out, self.num_filters), dtype = cp.float32)

        #create our sliding window
        self.patches = as_strided(
            padded_inputs,
            shape=(S, H_out, W_out, fH, fW, D_in),
            strides=(
                padded_inputs.strides[0],       # step between samples
                padded_inputs.strides[1] * sH,  # step down a row
                padded_inputs.strides[2] * sW,  # step across a column
                padded_inputs.strides[1],       # move down 1 row inside patch
                padded_inputs.strides[2],       # move right 1 col inside patch
                padded_inputs.strides[3],       # step across channels
            )
        )

        #Keep the samples, h_out, w_out, and the number of channels out. But, iterate over the patch(x, y) with channels c, and with the number of filters d
        self.output = cp.einsum('shwxyc,xycd->shwd', self.patches, self.filter_weights)
        self.output += self.biases.reshape((1, 1, 1, self.num_filters)) 

        self.inputs = inputs
        self.padded_inputs = padded_inputs
        return self.output
        #save the output tensor using self. for backpropogation

    def backward(self, dvalues):

        #extract dvalues dimensions
        S, H_out, W_out, C_out = dvalues.shape
        fH, fW, C_in, C_out = self.filter_weights.shape
        sH, sW = self.strides
        _, H_in, W_in, _ = self.inputs.shape 

        #Now we need to account for dbiases and dweights

        self.dbiases = cp.sum(dvalues, axis = [0, 1, 2])

        self.dweights = cp.tensordot(self.patches, dvalues, axes = ([0, 1, 2], [0, 1, 2]))

        dilated_H = (H_out - 1) * sH + 1
        dilated_W = (W_out - 1) * sW + 1 
        
        dvalues_dilated = cp.zeros(shape= (S, dilated_H, dilated_W, C_out), dtype= dvalues.dtype)
        # Inject values using step slices
        dvalues_dilated[:, ::sH, ::sW, :] = dvalues
        
        # padding 
        if self.padding == "same": 
            backward_pad_h = (fH - 1) - self.forward_pad_h
            backward_pad_w = (fW - 1) - self.forward_pad_w
        if self.padding == "valid": 
            backward_pad_h = (fH - 1)
            backward_pad_w = (fW - 1)
        
        dvalues_padded = cp.pad(dvalues_dilated, pad_width= ((0, 0), (backward_pad_h, backward_pad_h), (backward_pad_w, backward_pad_w), (0, 0)))

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
    
        self.dinputs = cp.tensordot(
            dvalues_patches, 
            flipped_weights, 
            axes=([3, 4, 5], [0, 1, 3]) # Match fH, fW, C_out
        )

        return self.dinputs