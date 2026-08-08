import numpy as np
import aether.config as config
from aether.base import Layer
class Conv(Layer):
    def __init__(self, in_channels, out_channels = 1, filter_size = (3, 3), stride = (1, 1), padding = "same"):

        xp = config.xp
        # input_shape has form (batch_size, height, width, channels)
        self.C_in = in_channels
        self.C_out = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding 
        self.biases = xp.zeros(self.C_out, dtype = xp.float32)
        self.weight_regularizer_l1 = 0
        self.weight_regularizer_l2 = 0
        self.bias_regularizer_l1 = 0
        self.bias_regularizer_l2 = 0

        # We'll handle two scenarios, the first, where we pass in a (n, n, 1) or grayscale image, and a second
        # where we'll handle a (n, n, 3) or RGB image. 
        n = self.filter_size[0] * self.filter_size[1] * self.C_in
        std = xp.sqrt(xp.float32(2.0 / n))
        
        # We can now do He initaliztion, we'll sample values from a standard distribution N (0, 1) and multiply it by our
        # std value to get N(0, std) 
        self.filter_weights = (xp.random.randn(
            filter_size[0],         # Filter height fH
            filter_size[1],         # Filter width fW
            self.C_in,              # Input channels C_in 
            self.C_out              # Output channels C_out
        ).astype(xp.float32)* std)

        self.weights = self.filter_weights

    def forward(self, inputs, training):
        
        xp = config.get_array_module(inputs)
        as_strided = config.get_stride_utility(xp)
        # Extract input dimensions
        fH, fW = self.filter_size
        sH, sW = self.stride
        S, H_in, W_in, D_in = inputs.shape
        
        # Creating padding depending on padding = 'same', or padding = 'valid'
        if self.padding == "same":
            self.forward_pad_h = (self.filter_size[0] - 1) // 2
            self.forward_pad_w = (self.filter_size[1] - 1) // 2
        else:            
            self.forward_pad_h = 0
            self.forward_pad_w = 0

        #We need integer output dimensions, so cast equations to int
        H_out = int((H_in + 2 * self.forward_pad_h - self.filter_size[0]) / self.stride[0] + 1)
        W_out = int((W_in + 2 * self.forward_pad_w - self.filter_size[1]) / self.stride[1] + 1)
        
        # (0, 0) -> don't touch the number of samples in the batch
        # (P, P) -> pad top and bottom pixels by P pixels (axis 1)
        # (P, P) -> pad left and right pixels by P pixels (axis 2)
        # (0, 0) -> don't pad depth. 
        # contstant -> add constant_values for the padded values
        padded_inputs = xp.pad(array = inputs, 
                            pad_width = ((0, 0), (self.forward_pad_h, self.forward_pad_h), (self.forward_pad_w, self.forward_pad_w), (0, 0)),
                            mode = 'constant',
                            constant_values = 0).astype(xp.float32, copy = False)

        #Create an output tensor of size (batch_size, H_out, W_out, C_out)
        self.output = xp.zeros((S, H_out, W_out, self.C_out), dtype = xp.float32)

        # Create our sliding window
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

        # Keep the samples, h_out, w_out, and C_out. 
        # But, iterate over the patch(x, y) with channels c, and with the number of filters d
        self.output = xp.einsum('shwxyc,xycd->shwd', self.patches, self.filter_weights, optimize=True)
        self.output += self.biases.reshape((1, 1, 1, self.C_out)) 

        self.inputs = inputs
        self.padded_inputs = padded_inputs
        return self.output
        #save the output tensor using self. for backpropogation

    
    def backward(self, dvalues):

        xp = config.get_array_module(dvalues)
        as_strided = config.get_stride_utility(xp)
        
        S, H_out, W_out, C_out = dvalues.shape
        fH, fW, C_in, C_out = self.filter_weights.shape
        sH, sW = self.stride
        _, H_in, W_in, _ = self.inputs.shape 

        # Now we need to account for dbiases and dweights

        self.dbiases = xp.sum(dvalues, axis = (0, 1, 2))

        self.dweights = xp.tensordot(self.patches, dvalues, axes = ([0, 1, 2], [0, 1, 2]))

        dilated_H = (H_out - 1) * sH + 1
        dilated_W = (W_out - 1) * sW + 1 
        
        dvalues_dilated = xp.zeros(shape= (S, dilated_H, dilated_W, C_out), dtype= dvalues.dtype)
        # Inject values using step slices
        dvalues_dilated[:, ::sH, ::sW, :] = dvalues
        
         
        if self.padding == "same": 
            backward_pad_h = (fH - 1) - self.forward_pad_h
            backward_pad_w = (fW - 1) - self.forward_pad_w
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
    def weights(self):
        return self.filter_weights

    @weights.setter
    def weights(self, value):
        self.filter_weights = value