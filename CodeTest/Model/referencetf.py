import time
import numpy as np
import tensorflow as tf
from nnfs.datasets import spiral_data

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

y = tf.keras.utils.to_categorical(y, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(512, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                          bias_regularizer=tf.keras.regularizers.l2(5e-4)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05, decay=5e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

class PrintEvery(tf.keras.callbacks.Callback):
    def __init__(self, print_every=500):
        super().__init__()
        self.print_every = print_every

    def on_epoch_end(self, epoch, logs=None):
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
    batch_size=len(X),
    epochs=5000,
    verbose=0,  # silent
    validation_data=(X_test, y_test),
    callbacks=[PrintEvery(print_every=500)]
)
end = time.time()

print(f"\nTensorFlow training time: {end - start:.2f} seconds")
