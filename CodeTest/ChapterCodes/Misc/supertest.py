import matplotlib.pyplot as plt

epochs = [0, 500, 1000, 1500, 2000, 2500, 3000, 
          3500, 4000, 4500, 5000, 5500, 6000, 
          6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]

accuracy = [0.360, 0.687, 0.837, 0.840, 0.897, 0.917, 0.923,
            0.940, 0.940, 0.940, 0.947, 0.947, 0.943,
            0.947, 0.947, 0.947, 0.950, 0.953, 0.957, 0.957, 0.957]

loss = [1.099, 0.666, 0.447, 0.395, 0.282, 0.222, 0.192,
        0.169, 0.158, 0.150, 0.145, 0.141, 0.138,
        0.135, 0.132, 0.130, 0.128, 0.126, 0.124, 0.123, 0.121]

plt.figure(figsize=(8,4))
plt.plot(epochs, accuracy, label="Accuracy", marker='o')
plt.plot(epochs, loss, label="Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.show()
