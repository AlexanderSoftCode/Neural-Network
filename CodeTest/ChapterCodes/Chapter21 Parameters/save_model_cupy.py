import numpy as np
import cupy as cp
import time
from Model.model_cupy import *
#Load dataset

#cp.savez("fashion_mnist_test_cupy.npz", X=X_test, y=y_test)
data = cp.load('fashion_mnist_train_cupy.npz')
X = data["X"]
y = data["y"]

data = cp.load('fashion_mnist_test_cupy.npz')
X_test = data["X"]
y_test = data["y"]

keys = cp.array(range(X.shape[0]))
cp.random.shuffle(keys)
X = X[keys]
y = y[keys]

#Scale and reshape
X = (X.reshape(X.shape[0], -1).astype(cp.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(cp.float32) - 127.5) / 127.5

model = Model()

model.add(Layer_Dense(X.shape[1], 256, weight_regularizer_l2=5e-4, bias_regularizer_l2= 5e-4))
model.add(ReLU())
model.add(Layer_Dense(256, 256, weight_regularizer_l2= 5e-4, bias_regularizer_l2= 5e-4))
model.add(ReLU())
model.add(Layer_Dense(256, 10, weight_regularizer_l2= 5e-4, bias_regularizer_l2= 5e-4))
model.add(SoftMax())

model.set(
    loss = Loss_CategoricalCrossEntropy(),
    optimizer= Optimizer_Adam(decay = 1e-4),
    accuracy= Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data= (X_test, y_test),
            epochs = 10, batch_size = 128, print_every = 500)


model.save('fashion_mnist.model_cupy')
#Once we save the model we will do a new file that will load the model. 
