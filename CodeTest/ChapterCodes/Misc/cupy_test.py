import numpy as np 
import cupy as cp 
import time 
x = cp.arange(10)
print(x) 

n_inputs = 10000
n_neurons = 12000
#print
start = time.time()
weights_cpu = .01 * np.random.randn(n_inputs, n_neurons)
end = time.time()
print("Numpy time ", end - start, " seconds")

start = time.time()
weights_gpu = 0.01 * cp.random.randn(n_inputs, n_neurons)
cp.cuda.Stream.null.synchronize()
end = time.time()
print("CuPy time ", end - start, " seconds")