import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2
import os

#this is to show an example of what one of the images look like. 
image_data = cv2.imread('fashion_mnist_images/train/4/0011.png',
                        cv2.IMREAD_UNCHANGED)

plt.imshow(image_data, cmap='gray')
plt.show()


#loads mnist
def load_mnist_dataset(dataset, path):
    #Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    X = []
    y = []

    for label in labels:
        #for each image in a given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            #read image
            image = cv2.imread(os.path.join(
                path, dataset, label, file
            ), cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)
    #convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test

#We can now load our data by doing
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
