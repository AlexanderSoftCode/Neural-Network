import numpy as np
from Model.modelfinal import *
import cv2

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

confidences = model.predict(X_test[:5])
print(confidences)

#We're calling Softmax.predict() here
predictions = model.output_layer_activation.predictions(confidences)
print(y_test[:5])

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser', 
    2: 'Pulloever',
    3: 'Dress', 
    4: 'Coat',
    5: 'Sandal', 
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

for prediction in predictions:
    print(fashion_mnist_labels[prediction])
