from Model.classes import Layer_Dense, ReLU, SoftMax, Loss_CategoricalCrossEntropy
import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
print('fortnite')
list = [[3, 2, 5],
        [9, 7, 1], 
        [4, 3, 6]]
np.array(list)
print(list)
#tuples are used with parenthesis
np.random.random((2,4)) #np.random numpy module, .random is function name
#we an use np.int64 and whatnot 
print(np.array([1.32, 5.78, 175.55]).dtype)

float32_array = np.array([1.32, 5.78, 175.55], dtype=np.float32)

inputs = [1.0, 2.0, 3.0, 2.5]

weights = [
    [0.2, 0.8, -0.5, 1.0],      #When we have [[4][4][4]] Then that means we have an array (big bracket)
    [0.5, -0.91, 0.26, -0.5],   #Then we have 3 rows aka number of first sub bracket
    [-0.26, -0.27, 0.17, 0.87]  #Then at the end when we count entries we see there are 4 per row, so 3x4 array in python.
]

biases = [2.0, 3.0, 0.5]
layer_outputs = np.dot(weights, inputs) + biases
#Basically, when you do nxm mnl operations we have to have the inner part of the operation be the aame number, else
#we get an error. 

print(layer_outputs)

a = [1,2,3]
b = [2,3,4] 

a = np.array([a])
b = np.array([b]).T 
inputs = [
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]]
weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases  #3x3 matrix which is why we have nx3 weights2
weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2 #This will give another 3x3 result.
print(layer2_outputs)

nnfs.init()       #Sets the random seed to 0 creates a float32 dtype, and uses dot product from np
X, y = spiral_data(samples = 100, classes = 3)
plt.scatter(X[:,0], X[:,1])
plt.show()


X, y = spiral_data(samples = 100, classes = 3)
dense1 = Layer_Dense(2,3)
activation1 = ReLU()
dense2 = Layer_Dense(3,3)
activation2 = SoftMax()
loss_function = Loss_CategoricalCrossEntropy()
dense1.forward(X) #x is our input
activation1.forward(dense1.layer_outputs) #so just pass the outputs from first layer
dense2.forward(activation1.output) #pass the activation layer outputs to 2nd layer
activation2.forward(dense2.layer_outputs)
loss = loss_function.forward(activation2.output, y) #book no give this
print(activation2.output[:5])
print('loss', loss)

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([0, 1, 1])
#Remember, we want to take the element with highest probability.
predicted_outputs = softmax_outputs[range(len(softmax_outputs)), class_targets]
negative_log = -np.log(predicted_outputs)
average_loss = np.mean(negative_log) #Remember the average is mean so yeah 
print(negative_log)
print(average_loss)
predictions = np.argmax(softmax_outputs, axis=1)
if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1) #convert the onehot
accuracy = np.mean(predictions == class_targets)
print('acc', accuracy)