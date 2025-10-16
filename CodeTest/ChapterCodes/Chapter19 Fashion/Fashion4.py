import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

X, y = spiral_data(samples = 100, classes = 3)
EPOCHS = 10
BATCH_SIZE = 128 #take 128 samples at a time

#Calculate the number of steps per epoch
steps = X.shape[0] // BATCH_SIZE
#Dividing rounds down. If there is any remaining data left but not a 
#full batch then add 1 to include into the remaining samples 
if steps * BATCH_SIZE < X.shape[0]:
    steps += 1

for epoch in range(EPOCHS):
    for step in range(steps):
        batch_x = X[BATCH_SIZE * step: (step+1)*BATCH_SIZE]
        batch_y = y[BATCH_SIZE * step: (step+1)*BATCH_SIZE]
        #Finally, we begin using our model. 
