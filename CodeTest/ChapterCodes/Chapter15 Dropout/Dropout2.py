import numpy as np
import nnfs
import matplotlib.pyplot as plt
#Classes.py
from Classes import *
from nnfs.datasets import spiral_data
nnfs.init() 

X, y = spiral_data(samples = 100, classes = 3)

#plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg", s=40, edgecolor='k')
#plt.title("Spiral Dataset (3 Classes)")
#plt.xlabel("Feature 1")
#plt.ylabel("Feature 2")
#plt.show()
dense1 = Layer_Dense(2, 64, weight_regularizer_l2= 5e-4,
                    bias_regularizer_l2= 5e-4) #2 inputs 64 weights
activation1 = ReLU()
dropout1 = Layer_Dropout(rate = 0.1)  #Success rate of .1 
dense2 = Layer_Dense(64, 3) #3 output values for output layer
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

#optimizer = Optimizer_SGD(decay = 1e-3, momentum = 0.9) 
optimizer = Optimizer_Adam(learning_rate = 0.05, decay = 5e-5)
for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    #expected output from layer2 vs real
    data_loss = loss_activation.forward(dense2.output, y) 

    regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
                          loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss
    #lets now calculate accuracy from output of activation 2 and targets
    #calc values along first axis
    predictions = np.argmax(loss_activation.output, axis = 1) 
    if len(y.shape) == 2:
        y = np.argmax(y, axis = 1) 
    accuracy = np.mean(predictions == y) 
    if not epoch % 500: 
        print(f'epoch: {epoch}, ' + 
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} ' +
              f'lr: {optimizer.current_learning_rate} ' +
              f'data_loss: {data_loss:.3f} ' +
              f'reg_loss: {regularization_loss:.3f} ')
    #Remember, given the raw output of layer2 we apply softmax first 
    loss_activation.backward(loss_activation.output, y) 
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs) 

    optimizer.pre_update_parameters()
    optimizer.update_parameters(dense1) #so we are going to take the negative 
    optimizer.update_parameters(dense2) #of our parameter
    optimizer.post_update_parameters()

    
X_test, y_test = spiral_data(samples = 100, classes = 3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)
predictions = np.argmax(loss_activation.output, axis = 1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis = 1)
accuracy = np.mean(predictions == y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')