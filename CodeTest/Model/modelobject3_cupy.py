import numpy as np
import cupy as cp
import nnfs
import time
from model_cupy import * 
nnfs.init()

start = time.time()
print("Cupy version")
data = np.load("fashion_mnist_train.npz")
X = cp.asarray(data["X"])  # Move to GPU
y = cp.asarray(data["y"])

# Save in CuPy-native format
cp.savez("fashion_mnist_train_cupy.npz", X=X, y=y)

data = np.load("fashion_mnist_test.npz")
X_test = cp.asarray(data["X"])
y_test = cp.asarray(data["y"])
cp.savez("fashion_mnist_test_cupy.npz", X=X_test, y=y_test)

#Now we need to shuffle the batches

keys = cp.array(range(X.shape[0]))
cp.random.shuffle(keys)
X = X[keys]
y = y[keys]

#Scale and reshape samples

X = (X.reshape(X.shape[0], -1).astype(cp.float32) - 127.5 ) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(cp.float32) -
            127.5) / 127.5

model = Model()
model.add(Layer_Dense(X.shape[1], n_neurons= 256, 
                      weight_regularizer_l2= 5e-4, bias_regularizer_l2= 5e-4))
model.add(ReLU())
model.add(Layer_Dense(256, 256, weight_regularizer_l2= 5e-4, bias_regularizer_l2= 5e-4))
model.add(ReLU())
model.add(Layer_Dense(256, 10))
model.add(SoftMax())

model.set(
    loss = Loss_CategoricalCrossEntropy(),
    optimizer = Optimizer_Adam(decay = 5e-5),
    accuracy = Accuracy_Categorical()
)

model.finalize()

cp.cuda.Stream.null.synchronize() 
model.train(X, y, validation_data = (X_test, y_test),
            epochs = 5, batch_size = 128, print_every = 1000)
cp.cuda.Stream.null.synchronize() 
end = time.time()
print(f"Total time elapsed {end - start:.2f} seconds")