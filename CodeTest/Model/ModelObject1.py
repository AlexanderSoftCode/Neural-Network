#classes.py
#model.py
from classes import *
from model import * 
from nnfs.datasets import sine_data

X, y = sine_data()

model = Model()

model.add(Layer_Dense(1, 64))
model.add(ReLU())
model.add(Layer_Dense(64, 64))
model.add(ReLU())
model.add(Layer_Dense(64,1))
model.add(Activation_Linear())

print(model.layers)

model.set(loss = Loss_MeanSquaredError(), 
          optimizer = Optimizer_Adam(learning_rate= 0.005, decay = 1e-3),
          accuracy = Accuracy_Regression())

model.finalize()

model.train(X = X, y = y, epochs = 10000, print_every = 500)