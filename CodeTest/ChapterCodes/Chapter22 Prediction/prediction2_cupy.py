import cupy as cp
from Model.model_cupy import *
import cv2
import matplotlib.pyplot as plt

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser', 
    2: 'Pulloever',
    3: 'Dress', 
    4: 'Coat',
    5: 'Sandal', 
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}
image_data = cv2.imread('tshirt.png', cv2.IMREAD_UNCHANGED)

#Open cv uses BGR so we need to use that over RGB
plt.imshow(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
plt.show()

#This time we get a grayscaled image
image_data = cv2.imread('tshirt.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(image_data, cmap = 'gray')
plt.show()

#Now we can resize

image_data = cv2.resize(image_data, (28, 28))
plt.imshow(image_data, cmap = 'gray')
plt.show()

image_data = 255 - image_data
image_data = (image_data.reshape(1, -1).astype(cp.float32) \
    - 127.5) / 127.5
image_data = cp.array(image_data)
model = Model.load('fashion_mnist.model_cupy')
confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)
prediction = fashion_mnist_labels[int(predictions[0])]
print(prediction)
