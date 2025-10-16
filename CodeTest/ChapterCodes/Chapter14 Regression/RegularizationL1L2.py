import numpy as np
import nnfs
#Classes.py
from Classes import *
from nnfs.datasets import spiral_data
nnfs.init() 

#the backward pass of l1 regularization is 2lambda * w(m) 
#the backward pass of l2 regularization is lambda(+- if w(m) <> 0)

weights = np.array([[.2, .8, -.5, .10000],
                    [.3, .4, -.26, -.1],
                    [.5, .8, -15, -.59]])               #for one sample
dL1 = np.ones_like(weights)
dL1[weights < 0] = -1
print(dL1) #all we have to do now is multiply this by lambda

X, y = spiral_data(samples = 100, classes = 3)
dense1 = Layer_Dense(2, 512, weight_regularizer_l2 = 5e-4, bias_regularizer_l2= 5e-4)
activation1 = ReLU()
dropout1 = Layer_Dropout(rate = 0.1) 
dense2 = Layer_Dense(512, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_Adam(learning_rate = .05, decay= 5e-5)

for epoch in range(10001): 
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
                          loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss 

    predictions = np.argmax(loss_activation.output, axis = 1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis = 1) 
    accuracy = np.mean(predictions == y) 

    if not epoch % 500:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')   
    #Time for backwards pass
    loss_activation.backward(loss_activation.output, y) 
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    #Lets now update the optimizer
    
    optimizer.pre_update_parameters()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update_parameters()

#Now lets try and update the model
X_test, y_test = spiral_data(samples = 100, classes = 3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
val_loss = loss_activation.forward(dense2.output, y_test)
predictions = np.argmax(loss_activation.output, axis = 1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis = 1)
val_accuracy = np.mean(predictions == y_test) 

print(f'Validation accuracy:  {val_accuracy:.3f}, loss: {val_loss:.3f}')