import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1 = 0,
                 bias_regularizer_l1 = 0, weight_regularizer_l2 = 0,
                 bias_regularizer_l2 = 0):
        #With He initalization, our fan_in maintains proper variance through layers.
        self.weights = .01 * np.random.randn(n_inputs, n_neurons) * \
            np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
    def forward(self, inputs, training):
        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)

        if self.weight_regularizer_l1 > 0:
             dL1 = np.ones_like(self.weights)
             dL1 [self.weights < 0] = -1
             self.dweights += self.weight_regularizer_l1 * dL1
        
        if self.weight_regularizer_l2 > 0:
             self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
             dL1 = np.ones_like(self.biases)
             dL1 [self.biases < 0 ] = -1
             self.dbiases += self.bias_regularizer_l1 * dL1 
        
        if self.bias_regularizer_l2 > 0:
             self.dbiases += 2* self.bias_regularizer_l2 * self.biases

        #Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Layer_Dropout:
    def __init__(self, rate):
        #We write rate as the success rate. The dropout rate will then be 
        self.rate = 1 - rate
    
    def forward(self, inputs, training):
        #were gonna save the inputs and the binary mask
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape) \
                        / self.rate
        self.output = self.binary_mask * self.inputs

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask 

#We do this only for completeness and clarity to see the activation
#function of the output layer in the model definition code.
class Activation_Linear:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = self.inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
    
    def predictions(self, outputs):
        return outputs

class ReLU:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] = 0 

class Sigmoid:  
    def forward(self, inputs, training): 
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)

class MeanSquaredError:

    def forward(self, y_pred, y_true):
        sample_loss = np.mean((y_true - y_pred)**2, axis = -1)
        return sample_loss
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0]) #same for all rows or should be

        self.dinputs = -2 * (y_true - dvalues) / outputs
        #Normalize
        self.dinputs = self.dinputs / samples
    
    def predictions(self, outputs):
        return (outputs > 0.5) * 1
class SoftMax:
    def forward(self, inputs, training):
        self.exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True)) #e**(inputs - max(inputs by row))
        probabilities = self.exp_values / np.sum(self.exp_values, axis=1, keepdims = True) #e**k / sum(e**k) 
        self.output = probabilities

    def backward(self, dvalues):                #Doing this function is expensive. If we combine loss and softmax we can get a simpler function. 
        self.dinputs = np.empty_like(dvalues) 

        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)): 
            #Flatten output array 
            single_output = single_output.reshape(-1, 1) 
            #Jacobian matrix
            jacobian = np.diagflat(single_output) - \
                       np.dot(single_output, single_output.T)
            #Get sample-wise gradient 
            self.dinputs[index] = np.dot(jacobian, single_dvalues)     

    def predictions(self, outputs):
        return np.argmax(outputs, axis = 1) #return the max of the rows
class Loss: 

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization= False):
        sample_losses = self.forward(output, y) #calc sample losses
        data_loss = np.mean(sample_losses)      #calc mean/average losses

        self.accumulated_sum += np.sum(sample_losses)
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
                                        np.sum(np.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                        np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                        np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                        np.sum(layer.biases * layer.biases) 
        return regularization_loss

class Loss_CategoricalCrossEntropy(Loss): 
    def forward(self, y_pred, y_true):
        #num samples in batch
        samples = len(y_pred)

        #next lets clip before continuing
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7) #.000001 -> .999999

        if len(y_true.shape) == 1:                                      #scale vector [0, 1, 2]
            correct_confidences = y_pred_clip[range(samples), y_true]
        elif len(y_true.shape) == 2:                                    #one hot encoding [0, 1, 0] [1, 0, 0]...
            correct_confidences = np.sum(y_pred_clip * y_true, axis=1)             #axis1 = sum rows, 
        neg_log_likelihoods = -np.log(correct_confidences)              #-log(0,0,0,.59,0,0,0)
        return neg_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        #number of labels per sample
        labels = len(dvalues[0]) 
        #if the labels are sparse turn them into one hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true] #create a lookup table of labelsxlabels with indexes y_true where y_true = 1xn 

        #calculate gradient 
        self.dinputs = -y_true / dvalues #we are dividng our true by softmax outputs. Then we get inputs. 
        #Normalize gradient with num samples
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossEntropy():
#    def __init__(self):
#        self.activation = SoftMax()
#        self.loss = Loss_CategoricalCrossEntropy()

    #y_true is the vector of correct class indices, one per sample.
    #dvalues is output of softmax layer shape(n_samples, n_classes)
#    def forward(self, inputs, y_true):
#        self.activation.forward(inputs)                 #call forward function of softmax
#        self.output = self.activation.output            #take the output as output of forward
#        return self.loss.calculate(self.output, y_true) #take the loss via the ouput of softmax versus true
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)                          #For the backward note the samples
        #If labels are one-hot encoded, 
        #turn them into discrete values
#        if len(y_true.shape) == 2:                      #if dataset answers return one hot
#            y_true = np.argmax(y_true, axis = 1)        #take the max of the rows

        self.dinputs = dvalues.copy() #copy 
        #subtracts 1 from the predicted probability of the correct class for each sample.
        #This turns the softmax outputs into the correct gradient expression
        # (softmax - one_hot) for backpropagation.
        self.dinputs[range(samples), y_true] -= 1
        #normalize
        self.dinputs = self.dinputs / samples 

# Mean Squared Error loss
class Loss_MeanSquaredError(Loss):  # L2 loss

    # Forward pass
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss):  # L1 loss

    # Forward pass
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        # Return losses
        return sample_losses


    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Optimizer_SGD: 
    #Initialize learning rate of 1
    def __init__(self, learning_rate = 1., decay = 0., momentum = 0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum 

    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
            (1. / (1. + self.decay * self.iterations))
            
    def update_parameters(self, layer):
        if self.momentum:
            #If there are no momentum arrays then create them
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)#If there is no momentum array for weigh)ts
                layer.bias_momentums = np.zeros_like(layer.biases)#The array doesn't exist for biases yet either. 
            #Build weight updates with momentum
            weight_updates = self.momentum * layer.weight_momentums - \
                             self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums - \
                           self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates
    #call once after any parameter updates
    def post_update_parameters(self):
        self.iterations += 1
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
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

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
            (np.sqrt(weight_cache_corrected) + self.epsilon)

        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)
    def post_update_parameters(self):
        self.iterations += 1
