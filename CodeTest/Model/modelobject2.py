#classes.py
#model.py
import time
from classes import *
from model import * 
from nnfs.datasets import spiral_data

start = time.time()
X, y = spiral_data(samples = 1000, classes = 3)
X_test , y_test = spiral_data(samples = 100, classes = 3)
 
model = Model()

model.add(Layer_Dense(n_inputs = 2, n_neurons = 512, weight_regularizer_l2 = 5e-4,
                      bias_regularizer_l2 = 5e-4))
model.add(ReLU())
model.add(Layer_Dropout(rate = 0.1))
model.add(Layer_Dense(n_inputs = 512, n_neurons = 3))
model.add(SoftMax())

model.set(
    loss = Loss_CategoricalCrossEntropy(),
    optimizer = Optimizer_Adam(learning_rate = 0.05, decay = 5e-5),
    accuracy = Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs = 5000, print_every = 250)
end = time.time()
print("Total time elapsed ", end - start , " seconds")