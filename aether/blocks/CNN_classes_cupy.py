import cupy as cp 
from cupy.lib.stride_tricks import as_strided
import aether.config as config

class Conv:
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
                            pad_width = ((0, 0), (self.forward_pad_h, self.forward_pad_h), (self.forward_pad_w, self.forward_pad_w), (0, 0)),
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
class Pooling:

    def __init__(self, filter_size = (2, 2), strides = (2, 2),
                  padding = "valid", pooling_type = "max"):
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.pooling_type = pooling_type
        
    def forward(self, inputs, training):
        #Inputs should be of shape (S, H_in, W_in, C = D_in) 
        inputs = inputs.astype(cp.float32, copy = False)
        if inputs.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, got {inputs.ndim} instead.")
        S, H_in, W_in, C = inputs.shape
        fH, fW = self.filter_size
        sH, sW = self.strides

        padding = self.padding
        if padding == "valid":
            H_out = int(cp.floor((H_in - fH) / sH + 1).item())
            W_out = int(cp.floor((W_in - fW) / sW + 1).item())

            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = 0, 0, 0, 0
        elif padding == "same":
            
            H_out = int(cp.ceil(H_in / sH).item())
            W_out = int(cp.ceil(W_in / sW).item())

            pad_h = max((H_out - 1) * sH + fH - H_in, 0)
            pad_w = max((W_out - 1) * sW + fW - W_in, 0)
            self.pad_top = pad_h // 2
            self.pad_bottom = pad_h - self.pad_top
            self.pad_left = pad_w // 2
            self.pad_right = pad_w - self.pad_left
            inputs = cp.pad(inputs, ((0,0), (self.pad_top,self.pad_bottom), (self.pad_left,self.pad_right), (0,0)), mode='constant')
        else: 
            raise ValueError(f"Expected padding == valid or same, recieved {padding} instead")

        #cast our output dimensions into ints from floats. 
        H_out, W_out = int(H_out), int(W_out)

        #create output tensor with new sizes
        self.output = cp.zeros(shape = (S, H_out, W_out, C))
        self.inputs = inputs
        patches = as_strided(
            inputs,
            shape = (S, H_out, W_out, fH, fW, C), 
            strides = (
                inputs.strides[0],      #step between samples
                inputs.strides[1] * sH, #step between rows
                inputs.strides[2] * sW, #step between columns
                inputs.strides[1],      #Move down 1 row inside patch
                inputs.strides[2],      #move right 1col inside patch
                inputs.strides[3],      #step between each channel
            )
        )

        if self.pooling_type == "max":
            pooled = patches.max(axis = (3, 4)) 
            #We'll reshape the window to become a 1d array of size fH * fW
            patches_reshaped = patches.reshape(S, H_out, W_out, fH * fW, C)
            flat_indicies = patches_reshaped.argmax(axis = 3)

            #Now, we'll convert those flat indicies back to row col coordinates withing each
            #(fH, fW) patch
            max_rows, max_cols = cp.unravel_index(flat_indicies, (fH, fW)) 
            self.max_indicies = (max_rows, max_cols) 
        
        elif self.pooling_type == "average":
            pooled = patches.mean(axis = (3, 4))
            
        #Store both of these for backprop
        self.inputs = inputs
        output = pooled
        return output
    
    def backward(self, dvalues):

        S, H_out, W_out, C = dvalues.shape
        fH, fW, sH, sW = self.filter_size, self.strides
        _, H_in, W_in, _ = self.inputs.shape

        if self.padding == "valid":
            pad_top = pad_bottom = pad_left = pad_right = 0
            padded_H, padded_W = H_in, W_in
        elif self.padding == "same":
                
            pad_top = self.pad_top
            pad_bottom = self.pad_bottom
            pad_left = self.pad_left
            pad_right = self.pad_right

            padded_H = H_in + pad_top + pad_bottom
            padded_W = W_in + pad_left + pad_right

        if self.pooling_type == "max":
            if self.strides == self.filter_size:    # Non-overlapping windows
                
                dinputs_padded = cp.zeros((S, padded_H, padded_W, C), dtype=dvalues.dtype)
                #compute non-overlapping windows
                inputs = self.inputs
                patches = as_strided(
                    inputs,
                    shape = (S, H_out, W_out, fH, fW, C), 
                    strides = (
                        inputs.strides[0],      # Step between samples
                        inputs.strides[1] * sH, # Step between rows
                        inputs.strides[2] * sW, # Step between columns
                        inputs.strides[1],      # Move down 1 row inside patch
                        inputs.strides[2],      # Move right 1 col inside patch
                        inputs.strides[3],      # Step between each channel
                    )
                )
                expanded_output = self.output[:, :, :, cp.newaxis, cp.newaxis, :]
                mask = (patches == expanded_output)

                #expand upstream gradient 
                expanded_dvalues = dvalues[:, :, :, cp.newaxis, cp.newaxis, :]
                dpatches = expanded_dvalues * mask

                dinputs_canvas = dpatches.transpose(0, 1, 3, 2, 4, 5).reshape(S, H_out * fH, W_out * fW, C)
                
                dinputs_padded[:, :H_out * fH, :W_out * fW, :] = dinputs_canvas
                dinputs = dinputs_padded[:, pad_top : pad_top + H_in, pad_left : pad_left + W_in, :]

            else:                                   # Branch where windows overlap
                
                max_rows, max_cols = self.max_indicies

                s_idx, h_idx, w_idx, c_idx = cp.ogrid[:S, :H_out, :W_out, :C]

                input_h = (h_idx * sH) + max_rows
                input_w = (w_idx * sW) + max_cols

                cp.add.at(dinputs_padded, (s_idx, input_h, input_w, c_idx), dvalues)

                dinputs = dinputs_padded[:, 
                            pad_top : pad_top + H_in, 
                            pad_left : pad_left + W_in, :]       
        if self.pooling_type == "average":
            if self.strides == self.filter_size: # Non-overlapping windows
                inputs = self.inputs
                patches = as_strided(inputs, 
                                    shape = (S, H_out, W_out, fH, fW, C),
                                    strides = (
                                        inputs.strides[0],      # Step between samples
                                        inputs.strides[1] * sH, # Step between rows
                                        inputs.strides[2] * sW, # Step between columns
                                        inputs.strides[1],      # Move down 1 row inside patch
                                        inputs.strides[2],      # Move right 1 col inside patch
                                        inputs.strides[3],      # Step between each channel
                    ))
                
                mask = cp.ones_like(patches)
                expanded_dvalues = dvalues[:, :, :, cp.newaxis, cp.newaxis, :]
                dpatches = (expanded_dvalues * mask) / (fH * fW)

                dinputs_padded = cp.zeros((S, padded_H, padded_W, C), dtype=dvalues.dtype)
                dinputs_canvas = dpatches.transpose(0, 1, 3, 2, 4, 5).reshape(S, H_out * fH, W_out * fW, C)
                dinputs_padded[:, :H_out * fH, :W_out * fW, :] = dinputs_canvas
                dinputs = dinputs_padded[:, pad_top : pad_top + H_in, pad_left : pad_left + W_in, :]
                
            else:                                   # Branch where windows overlap

                dilated_H = (H_out - 1) * sH + 1
                dilated_W = (W_out - 1) * sW + 1

                dvalues_dilated = cp.zeros(S, dilated_H, dilated_W, C, dtype=dvalues.dtype)

                backward_pad_top = (fH - 1) - pad_top
                backward_pad_left = (fW - 1) - pad_left
                backward_pad_bottom = (H_in + fH - 1) - dilated_H - backward_pad_top
                backward_pad_right = (W_in + fW - 1) - dilated_W - backward_pad_left

                dvalues_padded = cp.pad(dvalues_dilated, pad_width = (
                    (0, 0), (backward_pad_top, backward_pad_bottom), (backward_pad_left, backward_pad_right), (0, 0)
                ))

                dvalues_patches = as_strided(
                    dvalues_padded,
                    shape=(S, H_out, W_out, fH, fW, C),
                    strides=(
                        dvalues_padded.strides[0],       # step between samples
                        dvalues_padded.strides[1] * sH,  # step down a row
                        dvalues_padded.strides[2] * sW,  # step across a column
                        dvalues_padded.strides[1],       # move down 1 row inside patch
                        dvalues_padded.strides[2],       # move right 1 col inside patch
                        dvalues_padded.strides[3],       # step across channels
                    ))
                scale_factor = cp.array(1.0 / (fH * fW), dtype = dvalues.dtype)
                dinputs = dvalues_patches.sum(axis = (3,4) * scale_factor)

        return dinputs

class Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1 = 0,
                 bias_regularizer_l1 = 0, weight_regularizer_l2 = 0,
                 bias_regularizer_l2 = 0):
        #With He initalization, our fan_in maintains proper variance through layers.
        self.weights = .01 * cp.random.randn(n_inputs, n_neurons) * \
            cp.sqrt(2.0 / n_inputs)
        self.biases = cp.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
    def forward(self, inputs, training):
        self.inputs = inputs 
        self.output = cp.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        self.dweights = cp.dot(self.inputs.T, dvalues)
        self.dbiases = cp.sum(dvalues, axis = 0, keepdims = True)

        if self.weight_regularizer_l1 > 0:
             dL1 = cp.ones_like(self.weights)
             dL1 [self.weights < 0] = -1
             self.dweights += self.weight_regularizer_l1 * dL1
        
        if self.weight_regularizer_l2 > 0:
             self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
             dL1 = cp.ones_like(self.biases)
             dL1 [self.biases < 0 ] = -1
             self.dbiases += self.bias_regularizer_l1 * dL1 
        
        if self.bias_regularizer_l2 > 0:
             self.dbiases += 2* self.bias_regularizer_l2 * self.biases

        #Gradient on values
        self.dinputs = cp.dot(dvalues, self.weights.T)

    def get_parameters(self):
        #pass We'll let Model call this function 
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
        
class Dropout:
    def __init__(self, rate):
        #We write rate as the success rate. The dropout rate will then be 
        self.rate = 1 - rate
    
    def forward(self, inputs, training):
        #were gonna save the inputs and the binary mask
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return self.output
        self.binary_mask = cp.random.binomial(1, self.rate, size = inputs.shape) \
                        / self.rate
        self.output = self.binary_mask * self.inputs

        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask 

class SpatialDropout: 
    def __init__(self, rate):
        
        self.rate = rate
        self.keep_prob = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return self.output
        C = self.inputs.shape[-1]
        self.channel_mask = cp.random.binomial(1, self.keep_prob, size = (1, 1, 1, C)) \
                            / self.keep_prob
        self.output = inputs * self.channel_mask

        return self.output
    
    def backward(self, dvalues): 
        self.dinputs = dvalues * self.channel_mask

class ReLU:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = cp.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] = 0 

class Leaky_ReLU:
    def __init__(self, alpha = 0.01):
        self.alpha = alpha
    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = cp.where(inputs > 0, inputs, self.alpha * inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] *= self.alpha

class Batch_Norm:
    def __init__ (self, epsilon = 1e-5, momentum = 0.9, n_features = None):
        self.epsilon = epsilon
        self.momentum = momentum
        if n_features is not None:
            self.gamma = cp.ones(n_features, dtype=cp.float32)
            self.beta = cp.zeros(n_features, dtype=cp.float32)
        else:
            self.gamma = None
            self.beta = None

        self.weights = self.gamma
        self.biases = self.beta
        self.running_mean = None
        self.running_var = None
        self.weight_regularizer_l1 = 0
        self.weight_regularizer_l2 = 0
        self.bias_regularizer_l1 = 0
        self.bias_regularizer_l2 = 0

    def forward(self, inputs, training):
        self.inputs = inputs        

        if self.gamma is None: 
            C = inputs.shape[-1]
            self.gamma = cp.ones(C, dtype=cp.float32)
            self.beta = cp.zeros(C, dtype=cp.float32)
            self.running_mean = cp.zeros(C, dtype=cp.float32)
            self.running_var = cp.ones(C, dtype=cp.float32)
            self.weights = self.gamma
            self.biases = self.beta

        if inputs.ndim == 4: #if cnn
            axis = (0, 1, 2) 
        else: #dense
            axis = 0
        
        if training: 
            self.batch_mean = cp.mean(inputs, axis = axis, keepdims = True)
            self.batch_var = cp.var(inputs, axis = axis, keepdims = True)

            self.normalized = (inputs - self.batch_mean) / cp.sqrt(self.batch_var + self.epsilon)
            self.output = self.gamma * self.normalized + self.beta

            #now update the running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        
        else:
            self.normalized = (inputs - self.running_mean) / cp.sqrt(self.running_var + self.epsilon)
            self.output = self.gamma * self.normalized + self.beta

        return self.output 
    
    def backward(self, dvalues):
        axes = (0, 1, 2) if dvalues.ndim == 4 else (0,)
        N_total = cp.prod(cp.array([self.inputs.shape[ax] for ax in axes]))

        dhatx = dvalues * self.gamma # same shape as (N, H, W, C)

        dvar = cp.sum(dhatx * (self.inputs - self.batch_mean)
                    * (-0.5)
                    * cp.power(self.batch_var + self.epsilon, -1.5),
                    axis = axes,
                    keepdims = True)

        dmu = cp.sum(dhatx * (-1.0 / cp.sqrt(self.batch_var + self.epsilon)),
                    axis = axes, keepdims = True) \
                    + dvar * cp.sum(-2.0 * (self.inputs - self.batch_mean),
                    axis = axes, keepdims = True) / N_total

        inv_sqrt = 1.0 / cp.sqrt(self.batch_var + self.epsilon) #shape (1, 1, 1, C)
        self.dinputs = (dhatx * inv_sqrt + dvar * 2.0 * (self.inputs - self.batch_mean) / N_total \
                + dmu / N_total)
        
        self.dweights = cp.sum(dvalues * self.normalized, axis=axes)
        self.dbiases = cp.sum(dvalues, axis=axes)

        return self.dinputs 
    
    def get_parameters(self):
        return (
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var
        )

    def set_parameters(self, gamma, beta, running_mean, running_var):
        self.gamma = gamma
        self.beta = beta
        self.running_mean = running_mean
        self.running_var = running_var

        # keep internal references consistent
        self.weights = self.gamma
        self.biases  = self.beta

class Flatten:
    def forward(self, inputs, training):
        # Save shape so we can restore it in backward pass
        self.inputs_shape = inputs.shape
        # Flatten all dimensions except batch size
        self.output = inputs.reshape(inputs.shape[0], -1)

        return self.output
    
    def backward(self, dvalues):
        # Reshape gradients back to input shape
        self.dinputs = dvalues.reshape(self.inputs_shape)

class SoftMax:
    def forward(self, inputs, training):
        self.exp_values = cp.exp(inputs - cp.max(inputs, axis=1, keepdims = True)) #e**(inputs - max(inputs by row))
        probabilities = self.exp_values / cp.sum(self.exp_values, axis=1, keepdims = True) #e**k / sum(e**k) 
        self.output = probabilities

        return self.output

    def backward(self, dvalues):                #Doing this function is expensive. If we combine loss and softmax we can get a simpler function. 
        self.dinputs = cp.empty_like(dvalues) 

        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)): 
            #Flatten output array 
            single_output = single_output.reshape(-1, 1) 
            #Jacobian matrix
            jacobian = cp.diagflat(single_output) - \
                       cp.dot(single_output, single_output.T)
            #Get sample-wise gradient 
            self.dinputs[index] = cp.dot(jacobian, single_dvalues)     

    def predictions(self, outputs):
        return cp.argmax(outputs, axis = 1) #return the max of the rows
    
class Loss: 
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization= False, training = True):
        sample_losses = self.forward(output, y, training) #calc sample losses
        data_loss = cp.mean(sample_losses)      #calc mean/average losses

        self.accumulated_sum += cp.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    def calculate_accumulated(self, *, include_regularization = False):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss() 
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def regularization_loss(self):
        regularization_loss = 0             #if we don't do this, we risk overfitting.
                                            #We will have to denote partials for this too...
        for layer in self.trainable_layers:        
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                        cp.sum(cp.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                        cp.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                        cp.sum(cp.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                        cp.sum(layer.biases * layer.biases) 
        return regularization_loss

class Loss_CategoricalCrossEntropy(Loss): 
    def __init__(self, label_smoothing = 0.0):
        self.label_smoothing = label_smoothing 

    def forward(self, y_pred, y_true, training = True):
        #num samples in batch
        n_classes = y_pred.shape[1]
        #next lets clip before continuing
        y_pred_clip = cp.clip(y_pred, 1e-7, 1 - 1e-7) #.000001 -> .999999

        if len(y_true.shape) == 1:                    #scale vector [0, 1, 2]
            y_true = cp.eye(n_classes)[y_true]
            
        #apply label smoothing if used
        if self.label_smoothing > 0 and training:
            y_true = y_true * (1.0 - self.label_smoothing) + \
                    self.label_smoothing / n_classes
        
        #standard CE loss
        loss = -cp.sum(y_true * cp.log(y_pred_clip), axis = 1)
        
        return loss
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        n_classes = dvalues.shape[1]

        #number of labels per sample
        #if the labels are sparse turn them into one hot vector
        if len(y_true.shape) == 1:

            #create a lookup table of n_classesxnclasses
            # with indexes y_true where y_true = 1xn 
            y_true = cp.eye(n_classes)[y_true]

        #apply label smoothing again to match foward pass
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + \
                    self.label_smoothing / n_classes
            
        #calculate CE gradient
        self.dinputs = (dvalues - y_true) / samples 

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self, label_smoothing = 0.0):
        self.activation = SoftMax()
        self.loss = Loss_CategoricalCrossEntropy(label_smoothing)
        self.label_smoothing = label_smoothing
    #y_true is the vector of correct class indices, one per sample.
    #dvalues is output of softmax layer shape(n_samples, n_classes)
    def forward(self, inputs, y_true, training = True):
        self.activation.forward(inputs)                 #call forward function of softmax
        self.output = self.activation.output            #take the output as output of forward
        return self.loss.calculate(self.output, y_true, training = training) #take the loss via the ouput of softmax versus true
    
    def backward(self, dvalues, y_true, training = True):
        samples = len(dvalues)                          #For the backward note the samples
        n_classes = dvalues.shape[1]
        
        #if dataset is sparse, create one hot, 
        #else if one hot already then don't convert
        if len(y_true.shape) == 1:                      
            y_true = cp.eye(n_classes, dtype = cp.float32)[y_true]
        
        if self.label_smoothing > 0 and training:
            y_true = y_true * (1.0 - self.label_smoothing) \
                            + (self.label_smoothing / n_classes)

        #gradient = (p - y_smooth) where we normalize after 
        self.dinputs = (dvalues - y_true) / samples

#general starting learning rate for SGD is 1.0, with a decay down to 0.1. For Adam, a good starting 
#LR is 0.001 (1e-3), decaying down to 0.0001 (1e-4). Different problems may require different 
#values here, but these are decent to start.
class Optimizer_Adam:
    def __init__(self, learning_rate = .001, decay = 0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = .999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2 #used to be known as our rho 

    def pre_update_parameters(self):
        if self.decay:
            #self.learning_rate = initial learning rate. 1.0 / (1.0 * self.decay * self.iterations)
            #So this means that over time our current learning rate converges to 0 with the number of 
            #iterations
            self.current_learning_rate = self.learning_rate * \
            (1. / (1. + self.decay * self.iterations))
    def update_parameters(self, layer):
        if not hasattr(layer, "weight_cache"): #layer with column weight cache
            layer.weight_momentums = cp.zeros_like(layer.weights)
            layer.weight_cache = cp.zeros_like(layer.weights)
            layer.bias_momentums = cp.zeros_like(layer.biases)
            layer.bias_cache = cp.zeros_like(layer.biases)

        #self.beta_1 tends to zero once corrected
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
            (1- self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * (layer.dweights**2)
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * (layer.dbiases**2)
        
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1)) 
        
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
            (cp.sqrt(weight_cache_corrected) + self.epsilon)

        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
            (cp.sqrt(bias_cache_corrected) + self.epsilon)
    def post_update_parameters(self):
        self.iterations += 1