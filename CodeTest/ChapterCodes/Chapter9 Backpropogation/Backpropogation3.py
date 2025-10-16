import numpy as np
import nnfs
#Classes.py
from Classes import *
nnfs.init() 

X, y = spiral_data(samples = 100, classes = 3) 
dense1 = Layer_Dense(2, 3) 
activation1 = ReLU()
dense2 = Layer_Dense(3, 3) 
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function 
# takes the output of second dense layer here and returns loss 
loss = loss_activation.forward(dense2.output, y)
print(loss_activation.output[:5]) #first 5 entries
print("loss:", loss)

# Calculate accuracy from output of activation2 and targets 
# calculate values along first axis

predictions = np.argmax(loss_activation.output, axis = 1) #argmax is simple here
if len(y.shape) == 2:
    y = np.argmax(y, axis = 1) 
accuracy = np.mean(predictions == y) 
print("acc:", accuracy)

loss_activation.backward(loss_activation.output, y) 
dense2.backward(loss_activation.dinputs)  #remember derivative of inputs
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)
