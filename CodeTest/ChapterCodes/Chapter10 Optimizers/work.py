# Classes.py
from Classes import * 
import numpy as np 
import matplotlib.pyplot as plt
import nnfs     
from nnfs.datasets import vertical_data

nnfs.init()     

X, y = vertical_data(samples = 100, classes = 3) 
plt.scatter(X[:, 0], X[:,1], c=y, s= 40, cmap = 'brg')
plt.show()

dense1 = Layer_Dense(2,3)
activation1 = ReLU()
dense2 = Layer_Dense(3,3)
activation2 = SoftMax()
loss_function = Loss_CategoricalCrossEntropy()

#helper variables
lowest_loss = 999999 #some big number will be replaced soon
best_dense1_weights = dense1.weights.copy() #.copy for a full copy instead of 
best_dense1_biases = dense1.biases.copy()   #reference copy
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
    dense1.weights += .05 * np.random.randn(2, 3) #If no += we remove nudge and redo random
    dense1.biases += .05 * np.random.randn(1, 3) 
    dense2.weights += .05 * np.random.randn(3, 3) #numbers mean the dimensions 
    dense2.biases += .05 * np.random.randn(1, 3) 
    dense1.forward(X) 
    activation1.forward(dense1.layer_outputs)
    dense2.forward(activation1.output)
    activation2.forward(dense2.layer_outputs)

    loss = loss_function.calculate(activation2.output, y)  #y is our correct values. 
    predictions = np.argmax(activation2.output, axis = 1)  #lets get accuracy of our model 
    pred_average = np.mean(predictions==y)

    if(loss < lowest_loss):
        print("New set of weights found, iteration: " , iteration, 
              "loss:", loss, "acc", pred_average)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy() 
        lowest_loss = loss 
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
    #After running this code, while we get a better answer, now we fall under the local minimum. 

def f(x):
    return 2*x**2 
x = np.array(np.arange(0, 5, .001)) #like 50k x values I think
y = f(x) 

plt.plot(x, y) 
colors = ['k', 'g', 'r', 'b', 'c']

def approximate_tangent_line(x, approximate_derivative):
    return (approximate_derivative * x) + b
for i in range(5): 
    p2_delta = .0001
    x1 = i  
    x2 = x1 + p2_delta
    y1 = f(x1) #these are points and not lines
    y2 = f(x2) 
    
    print((x1, y1) , (x2, y2)) #our two points
    approximate_derivative = (y2-y1)/(x2-x1)  #gives us instantanious slope
    b = y2 - (approximate_derivative * x2) #instantanious yintercept of seconde point 
    to_plot = [x1-0.9, x1, x1+0.9]  #how much of that curve we want to plot and ranges. 
    plt.scatter(x1, y1, c = colors[i]) 
    plt.plot([point for point in to_plot],
             [approximate_tangent_line(point, approximate_derivative)
              for point in to_plot],
              c = colors[i])
    print('Approximate derivative for f(x)', 
          f'where x = {x1} is {approximate_derivative}')
plt.show()
