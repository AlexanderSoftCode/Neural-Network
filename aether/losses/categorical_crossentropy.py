import numpy as np
import aether.config as config
# Using Python 3.15 PEP 810: Explicit lazy imports would allow 
# for the import below to be called lazily, however for best 
# compatability this project would not use Python 3.15 for this issue
from aether.layers.activations import SoftMax
class Loss: 
    def __init__(self):
        self.new_pass()

    def get_fused_loss(self, last_layer):
        """Returns a fused activation+loss object if supported, else None"""
        return None
    
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization= False, training = True):
        xp = config.get_array_module(output)
        sample_losses = self.forward(output, y, training) #calc sample losses
        data_loss = xp.mean(sample_losses)      #calc mean/average losses

        self.accumulated_sum += xp.sum(sample_losses)
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
            xp = config.get_array_module(layer.weights)
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                        xp.sum(xp.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                        xp.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                        xp.sum(xp.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                        xp.sum(layer.biases * layer.biases) 
        return regularization_loss

class CategoricalCrossEntropy(Loss): 
    def __init__(self, label_smoothing = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing 

    def forward(self, y_pred, y_true, training = True):
        xp = config.get_array_module(y_pred)
        #num samples in batch
        n_classes = y_pred.shape[1]
        #next lets clip before continuing
        y_pred_clip = xp.clip(y_pred, 1e-7, 1 - 1e-7) #.000001 -> .999999

        if len(y_true.shape) == 1:                    #scale vector [0, 1, 2]
            y_true = xp.eye(n_classes)[y_true]
            
        #apply label smoothing if used
        if self.label_smoothing > 0 and training:
            y_true = y_true * (1.0 - self.label_smoothing) + \
                    self.label_smoothing / n_classes
        
        #standard CE loss
        loss = -xp.sum(y_true * xp.log(y_pred_clip), axis = 1)
        
        return loss
    
    def backward(self, dvalues, y_true, training=True):
        xp = config.get_array_module(dvalues)
        samples = len(dvalues)
        n_classes = dvalues.shape[1]

        #number of labels per sample
        #if the labels are sparse turn them into one hot vector
        if len(y_true.shape) == 1:

            #create a lookup table of n_classesxnclasses
            # with indexes y_true where y_true = 1xn 
            y_true = xp.eye(n_classes)[y_true]

        #apply label smoothing again to match foward pass
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + \
                    self.label_smoothing / n_classes
            
        dvalues_clip = xp.clip(dvalues, 1e-7, 1 - 1e-7)    
        #calculate CE gradient
        self.dinputs = -y_true / dvalues_clip / samples 

class SoftmaxCategoricalCrossEntropy(Loss):
    def __init__(self, label_smoothing = 0.0):
        self.activation = SoftMax()
        self.loss = CategoricalCrossEntropy(label_smoothing)
        self.label_smoothing = label_smoothing
        super().__init__()

    def get_fused_loss(self, last_layer):

        if isinstance(last_layer, SoftMax):
            return SoftmaxCategoricalCrossEntropy(label_smoothing=self.label_smoothing)
        return None
    
    def new_pass(self):
        super().new_pass()
        self.loss.new_pass()
    #y_true is the vector of correct class indices, one per sample.
    #dvalues is output of softmax layer shape(n_samples, n_classes)
    def forward(self, inputs, y_true, training = True):
        self.activation.forward(inputs, training=training)                 #call forward function of softmax
        self.output = self.activation.output            #take the output as output of forward
        return self.loss.calculate(self.output, y_true, training = training) #take the loss via the ouput of softmax versus true
    
    def backward(self, dvalues, y_true, training = True):
        xp = config.get_array_module(dvalues)
        samples = len(dvalues)                          #For the backward note the samples
        n_classes = dvalues.shape[1]
        
        #if dataset is sparse, create one hot, 
        #else if one hot already then don't convert
        if len(y_true.shape) == 1:                      
            y_true = xp.eye(n_classes, dtype = xp.float32)[y_true]
        
        if self.label_smoothing > 0 and training:
            y_true = y_true * (1.0 - self.label_smoothing) \
                            + (self.label_smoothing / n_classes)

        #gradient = (p - y_smooth) where we normalize after 
        self.dinputs = (dvalues - y_true) / samples