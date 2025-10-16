#from Model import classes
#from Model import model 
import numpy as np
np.set_printoptions(linewidth = 200)
import os
import cv2
import matplotlib.pyplot as plt
#C:\Users\Alex\Documents\fashion_mnist_images_L
BASE_PATH = r"C:\Users\Alex\Documents\fashion_mnist_images_L"
labels = os.listdir(os.path.join(BASE_PATH, "train"))
print(labels)
#[​'0'​, ​ '1'​, ​ '2'​, ​ '3'​, ​ '4'​, ​ '5'​, ​ '6'​, ​ '7'​, ​ '8'​, ​ '9'​] 

files = os.listdir(os.path.join(BASE_PATH, "train", "0"))
print(files[:10])
print(len(files))
image_data = cv2.imread(os.path.join(BASE_PATH, "train", "7", "0002.png"),
                        cv2.IMREAD_UNCHANGED)
print(image_data)
plt.imshow(image_data)
plt.show()

#Scan all the directories and create a list of lables
labels = os.listdir(os.path.join(BASE_PATH, "train"))
X = []
y = []

#For each label in folders
print(labels)
images_loaded = 0
for label in labels:
    for file in os.listdir(os.path.join(BASE_PATH, "train", label)):
        image = cv2.imread(os.path.join(BASE_PATH, "train", label, file), cv2.IMREAD_UNCHANGED)
        if image is not None:
            X.append(image)
            y.append(label)
        
        else:
            print(f"Warning: Could not load {file} in {label}")
        images_loaded += 1
        if images_loaded % 500 == 0:
            print(f"{images_loaded} images loaded so far...")

    print("I am now in label", label)