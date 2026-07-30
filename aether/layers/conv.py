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
        
        #(0, 0) -> don't touch the number of samples in the batch
        #(P, P) -> pad top and bottom pixels by P pixels (axis 1)
        #(P, P) -> pad left and right pixels by P pixels (axis 2)
        #(0, 0) -> don't pad depth. 
        #contstant -> add constant_values for the padded values
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

        # Keep the samples, h_out, w_out, and the number of channels out. 
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
    
class Pooling(Layer):

    def __init__(self, filter_size = (2, 2), strides = (2, 2),
                  padding = "valid", pooling_type = "max"):

        self.filter_size = filter_size
        self.stride = strides
        self.padding = padding
        self.pooling_type = pooling_type
        
    def forward(self, inputs, training):

        xp = config.get_array_module(inputs)
        as_strided = config.get_stride_utility(xp)
        #Inputs should be of shape (S, H_in, W_in, C = D_in) 
        if inputs.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, got {inputs.ndim} instead.")
        self.inputs = inputs
        S, H_in, W_in, C = inputs.shape
        fH, fW = self.filter_size
        sH, sW = self.stride

        padding = self.padding
        if padding == "valid":
            H_out = int(xp.floor((H_in - fH) / sH + 1).item())
            W_out = int(xp.floor((W_in - fW) / sW + 1).item())

            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = 0, 0, 0, 0
            inputs_padded = inputs
        elif padding == "same":
            
            H_out = int(xp.ceil(H_in / sH).item())
            W_out = int(xp.ceil(W_in / sW).item())

            pad_h = max((H_out - 1) * sH + fH - H_in, 0)
            pad_w = max((W_out - 1) * sW + fW - W_in, 0)
            self.pad_top = pad_h // 2
            self.pad_bottom = pad_h - self.pad_top
            self.pad_left = pad_w // 2
            self.pad_right = pad_w - self.pad_left
            inputs_padded = xp.pad(inputs, ((0,0), (self.pad_top,self.pad_bottom), (self.pad_left,self.pad_right), (0,0)))
        else: 
            raise ValueError(f"Expected padding == valid or same, recieved {padding} instead")

        #cast our output dimensions into ints from floats. 
        H_out, W_out = int(H_out), int(W_out)

        #create output tensor with new sizes
        self.output = xp.zeros(shape = (S, H_out, W_out, C))
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

        if self.pooling_type == "max":
            pooled = patches.max(axis = (3, 4)) 
            #We'll reshape the window to become a 1d array of size fH * fW
            patches_reshaped = patches.reshape(S, H_out, W_out, fH * fW, C)
            flat_indicies = patches_reshaped.argmax(axis = 3)

            #Now, we'll convert those flat indicies back to row col coordinates withing each
            #(fH, fW) patch
            max_rows, max_cols = xp.unravel_index(flat_indicies, (fH, fW)) 
            self.max_indicies = (max_rows, max_cols) 
        
        elif self.pooling_type == "average":
            pooled = patches.mean(axis = (3, 4))

        self.output = pooled
        return self.output
    
    def backward(self, dvalues):

        xp = config.get_array_module(dvalues)
        as_strided = config.get_stride_utility(xp)
        S, H_out, W_out, C = dvalues.shape
        fH, fW = self.filter_size
        sH, sW = self.stride
        _, H_in, W_in, _ = self.inputs.shape

        if self.padding == "valid":
            pad_top = pad_bottom = pad_left = pad_right = 0
            padded_H, padded_W = H_in, W_in
            inputs_padded = self.inputs
        else:  #padding == "same": 
                
            pad_top = self.pad_top
            pad_bottom = self.pad_bottom
            pad_left = self.pad_left
            pad_right = self.pad_right

            padded_H = H_in + pad_top + pad_bottom
            padded_W = W_in + pad_left + pad_right
            inputs_padded = xp.pad(self.inputs, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))

        if self.pooling_type == "max":
            if self.stride == self.filter_size:    # Non-overlapping windows
                
                dinputs_padded = xp.zeros((S, padded_H, padded_W, C), dtype=dvalues.dtype)
                #compute non-overlapping windows
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
                expanded_output = self.output[:, :, :, xp.newaxis, xp.newaxis, :]
                mask = (patches == expanded_output)

                #expand upstream gradient 
                expanded_dvalues = dvalues[:, :, :, xp.newaxis, xp.newaxis, :]
                dpatches = expanded_dvalues * mask

                dinputs_canvas = dpatches.transpose(0, 1, 3, 2, 4, 5).reshape(S, H_out * fH, W_out * fW, C)
                
                dinputs_padded[:, :H_out * fH, :W_out * fW, :] = dinputs_canvas
                self.dinputs = dinputs_padded[:, pad_top : pad_top + H_in, pad_left : pad_left + W_in, :]

            else:                                   # Branch where windows overlap
                
                max_rows, max_cols = self.max_indicies
                dinputs_padded = xp.zeros((S, padded_H, padded_W, C), dtype=dvalues.dtype)
                s_idx, h_idx, w_idx, c_idx = xp.ogrid[:S, :H_out, :W_out, :C]

                input_h = (h_idx * sH) + max_rows
                input_w = (w_idx * sW) + max_cols

                xp.add.at(dinputs_padded, (s_idx, input_h, input_w, c_idx), dvalues)

                self.dinputs = dinputs_padded[:, 
                            pad_top : pad_top + H_in, 
                            pad_left : pad_left + W_in, :]       
        if self.pooling_type == "average":
            if self.stride == self.filter_size: # Non-overlapping windows

                scaled_dvalues = dvalues / (fH * fW)
                # expand spatial height and width by fH and fW
                dinputs_canvas = xp.repeat(xp.repeat(scaled_dvalues, fH, axis=1), fW, axis=2)
                
                dinputs_padded = xp.zeros((S, padded_H, padded_W, C), dtype=dvalues.dtype)
                dinputs_padded[:, :H_out * fH, :W_out * fW, :] = dinputs_canvas
                self.dinputs = dinputs_padded[:, pad_top : pad_top + H_in, pad_left : pad_left + W_in, :]
                
            else:                                   # Branch where windows overlap

                dilated_H = (H_out - 1) * sH + 1
                dilated_W = (W_out - 1) * sW + 1

                dvalues_dilated = xp.zeros((S, dilated_H, dilated_W, C), dtype=dvalues.dtype)                
                dvalues_dilated[:, ::sH, ::sW, :] = dvalues
                
                backward_pad_top = (fH - 1) - pad_top
                backward_pad_left = (fW - 1) - pad_left
                backward_pad_bottom = (H_in + pad_top + self.pad_bottom + fH - 1) - dilated_H - backward_pad_top
                backward_pad_right = (W_in + pad_left + self.pad_right + fW - 1) - dilated_W - backward_pad_left
                dvalues_padded = xp.pad(dvalues_dilated, pad_width = (
                    (0, 0), (backward_pad_top, backward_pad_bottom), (backward_pad_left, backward_pad_right), (0, 0)
                ))

                dvalues_patches = as_strided(
                    dvalues_padded,
                    shape=(S, padded_H, padded_W, fH, fW, C),
                    strides=(
                        dvalues_padded.strides[0],       # step between samples
                        dvalues_padded.strides[1],       # step down a padded row
                        dvalues_padded.strides[2],       # step across a padded column 
                        dvalues_padded.strides[1],       # move down 1 row inside patch
                        dvalues_padded.strides[2],       # move right 1 col inside patch
                        dvalues_padded.strides[3],       # step across channels
                    ))
                dinputs_padded = dvalues_patches.sum(axis=(3, 4)) * (1.0 / (fH * fW))
                self.dinputs = dinputs_padded[
                    :, 
                    pad_top : pad_top + H_in, 
                    pad_left : pad_left + W_in, 
                    :
                ]

        return self.dinputs
