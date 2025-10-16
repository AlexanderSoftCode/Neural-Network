import time
import numpy as np
import tensorflow as tf

# Load data
data = np.load("fashion_mnist_train.npz")
X, y = data["X"], data["y"]

data = np.load("fashion_mnist_test.npz")
X_test, y_test = data["X"], data["y"]

# Reshape and scale like in your custom model
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

# One-hot encode (10 classes for Fashion-MNIST)
y = tf.keras.utils.to_categorical(y, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Build TensorFlow model to match your architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(256, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                          bias_regularizer=tf.keras.regularizers.l2(5e-4)),
    tf.keras.layers.Dense(256, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                          bias_regularizer=tf.keras.regularizers.l2(5e-4)),
    tf.keras.layers.Dense(10, activation="softmax")
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=5e-5)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

class PrintEvery(tf.keras.callbacks.Callback):
    def __init__(self, print_every=1):
        super().__init__()
        self.print_every = print_every

    def on_epoch_end(self, epoch, logs=None):
        # TensorFlow gives 0-based epoch
        if (epoch + 1) % self.print_every == 0:
            print(
                f"epoch: {epoch+1}, "
                f"acc: {logs['accuracy']:.3f}, "
                f"loss: {logs['loss']:.3f}, "
                f"val_acc: {logs['val_accuracy']:.3f}, "
                f"val_loss: {logs['val_loss']:.3f}"
            )

start = time.time()
history = model.fit(
    X, y,
    batch_size=128,     #  Match your custom model
    epochs=5,           #  Match your custom model
    verbose=0,          # Silent
    validation_data=(X_test, y_test),
    callbacks=[PrintEvery(print_every=1)]
)
end = time.time()

print(f"\nTensorFlow training time: {end - start:.2f} seconds")
