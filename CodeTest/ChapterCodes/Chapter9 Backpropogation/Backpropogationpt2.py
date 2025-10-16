import numpy as np
import nnfs
#Classes.py
from Classes import *
nnfs.init() 
#Gradient from next layer (think to the right).
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])
# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed so shape = (4,3)
weights = np.array([
    [0.2,   0.8,  -0.5,  1.0],          #Weights to 1 input 
    [0.5,  -0.91,  0.26, -0.5],         #Multiply row by dvalues. 
    [-0.26, -0.27, 0.17, 0.87]]).T

# One bias for each neuron 
# biases are the row vector with a shape (1, neurons)  
biases = np.array ([[2, 3, 0.5]])
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

#ReLU activation - derivative wrt input values from next layer passed to current layer 
drelu = relu_outputs.copy()
drelu[layer_outputs > 0] = 1    

dinputs = np.dot(drelu, weights.T) #to get dinputs multiply by weights
dweights = np.dot(inputs.T, drelu) #get dweights multiply by inputs
dbiases = np.sum(drelu, axis = 0, keepdims = True)

#update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases
print(weights)
print(biases)
#For each sample, compute gradient wrt inputs: dot product of
#upstream gradients (dvalues) with transposed weights
#dinputs = np.dot(dvalues, weights.T) 
#our gradient we are porducing
#backprop: dX = dZ * W**T. 
#Tranpose due to how weights are arranged in forward pass. 

softmax_outputs = np.array([
    [0.7, 0.1, 0.2],   # sample 1 probabilities
    [0.1, 0.5, 0.4],   # sample 2 probabilities
    [0.02, 0.9, 0.08]  # sample 3 probabilities
]) 
class_targets = np.array([0, 1, 1]) #our y true
softmax_loss = Activation_Softmax_Loss_CategoricalCrossEntropy()
softmax_loss.backward(softmax_outputs, class_targets)
dvalue1 = softmax_loss.dinputs         #(softmax outputs - one hot labels) / n samples
print(dvalue1)
