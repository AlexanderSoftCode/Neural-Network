import numpy as np
from Model.modelfinal import *
#Load dataset

data = np.load("fashion_mnist_train.npz")
X, y = data["X"], data["y"]

data = np.load("fashion_mnist_test.npz")
X_test, y_test = data["X"], data["y"]

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

#Scale and reshape
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

model = Model.load('fashion_mnist.model')

model.evaluate(X_test, y_test)
