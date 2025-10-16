import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2
import os

#loads mnist
def load_mnist_dataset(dataset, path, save_as):

    dataset_path = os.path.join(path, dataset)

    with os.scandir(dataset_path) as it:
        labels = [entry.name for entry in it if entry.is_dir()]
    labels.sort(key=int) #make sure 0..9 order is consistent

    num_images = sum(
        sum(1 for _ in os.scandir(os.path.join(dataset_path, label)))
        for label in labels
    )

    X = np.zeros((num_images, 28, 28), dtype = np.uint8)
    y = np.zeros(num_images, dtype = np.uint8)

    idx = 0
    for label in labels:
        label_path = os.path.join(dataset_path, label)
        with os.scandir(label_path) as it:
            for entry in it:
                image = cv2.imread(entry.path, cv2.IMREAD_UNCHANGED)
                X[idx] = image
                y[idx] = int(label)
                idx += 1
                if idx % 500 == 0:
                    print(f"{idx}/{num_images} images loaded so far...")
        print("I am now in label, ", label)
        
    np.savez_compressed(save_as, X = X, y = y)
    print(f"Saved dataset to {save_as}.npz")
    return X, y

def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path, save_as = "fashion_mnist_train")
    X_test, y_test = load_mnist_dataset('test', path, save_as = "fashion_mnist_test")

    return X, y, X_test, y_test

#We can now load our data by doing
BASE_PATH = r"C:\Users\Alex\Documents\fashion_mnist_images_L"

#this is to show an example of what one of the images look like. 
#image_data = cv2.imread(os.path.join(BASE_PATH, "train", "4", "0011.png"),
#                        cv2.IMREAD_UNCHANGED)
#plt.imshow(image_data, cmap='gray')
#plt.show()

#The below line should only be ran once, since we don't need
#to reload the dataset over and over again. 
#X, y, X_test, y_test = create_data_mnist(BASE_PATH)

data = np.load("fashion_mnist_train.npz")
X, y = data["X"], data["y"]

data = np.load("fashion_mnist_test.npz")
X_test, y_test = data["X"], data["y"]

X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
print(X.min(), X.max())
print(X.shape)
X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
X_test = X_test.reshape(X_test.shape[0], -1) 

keys = np.array(range(X.shape[0]))
print(keys[:10])
np.random.shuffle(keys)
print(keys[:10]) #Instead of printing say 0 1 2 3 4 5 ... its 15833 ...

X = X[keys]
y = y[keys]
#This tells NumPy to return values of given indices as we would normally index NumPy arrays, 
#but we’re using a variable that keeps a list of randomly ordered indices. Then we can check a slice 
#of targets:

print(y[15]) #So in this case, the 15th sample has class label 7 meaning we've shuffled...
BATCH_SIZE = 128
steps = X.shape[0] // BATCH_SIZE 

#If there is a straggling step, add them to the batch size. 
if steps * BATCH_SIZE < X.shape[0]:
    steps += 1
    
