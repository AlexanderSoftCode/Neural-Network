import numpy as np 
from Model.modelfinal import * 

#X, y, X_test, y_test = create_mnist_dataset("FILENAME")
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

model = Model()

model.add(Layer_Dense(X.shape[1], 128))
model.add(ReLU())
model.add(Layer_Dense(128, 128))
model.add(ReLU())
model.add(Layer_Dense(128, 10))
model.add(SoftMax())

model.set(
    loss = Loss_CategoricalCrossEntropy(),
    optimizer= Optimizer_Adam(decay = 1e-4),
    accuracy= Accuracy_Categorical()
)

model.finalize()

#model.train(X, y, validation_data= (X_test, y_test),
#            epochs = 10, batch_size = 128, print_every = 500)

#model.save_parameters('fashion_mnist.parms')
#so this one will save at CodeTest/fashion_mnist.parms
#This time we'll load the paramters first then we'll evaluate based on those paramters

#Now we are going to load the parameters
model.load_paramters('fashion_mnist.parms')

model.evaluate(X_test, y_test)