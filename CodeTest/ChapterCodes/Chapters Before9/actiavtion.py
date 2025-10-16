import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

nnfs.init
class Layer_Dense: 
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(inputs, 0) #If >0 keep, else set to 0 
class Activation_SoftMax:
    def forward(self, inputs):
        self.unormalized = np.exp(inputs - np.max(inputs, axis=1, keepdims= True))
        self.output = self.unormalized / np.sum(self.unormalized, axis= 1, keepdims= True)

X, y = spiral_data(samples = 100, classes = 3) #3 rows of inputs basically
dense1 = Layer_Dense(2, 3)     #2 inputs, 3 outputs our constructor.
activation1 = Activation_ReLU() #nothing happens but we still need an object.
dense2 = Layer_Dense(3, 3)     #3 inputs because of dense1, 3 outputs
activation2 = Activation_SoftMax()
dense1.forward(X)
activation1.forward(dense1.output) #output gets updated when we do .foward
dense2.forward(activation1.output) #Obviously the whole point was to use act f
activation2.forward(dense2.output)

print(activation2.output[0:5])