import numpy as np
import nnfs
import time
from model import * 
nnfs.init()

start = time.time()
print("Numpy version")
#X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
data = np.load("fashion_mnist_train.npz")
X, y = data["X"], data["y"]

data = np.load("fashion_mnist_test.npz")
X_test, y_test = data["X"], data["y"]

#Now we need to shuffle the batches

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

#Scale and reshape samples

X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5 ) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
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
model.train(X, y, validation_data = (X_test, y_test),
            epochs = 5, batch_size = 128, print_every = 1000)

# Forward pass through entire test dataset
outputs = model.forward(X_test, training=False)   # Already softmax probabilities

# Convert to predicted class indices
predictions = model.output_layer_activation.predictions(outputs)  # or: np.argmax(outputs, axis=1)

# Find incorrect predictions
misclassified = np.where(predictions != y_test)[0]
correct_class_0 = np.where((predictions == y_test) & (y_test == 0))[0]


end = time.time()
print(f"Total time elapsed {end - start:.2f} seconds")