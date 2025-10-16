# Classes.py
from Classes import * 
import numpy as np 

#forward pass 
x = [1.0, -2.0, 3.0] #input
w = [-3.0, -1.0, 2.0] #weights
b = 1.0 #bias

xw0 = x[0] * w[0]
xw1 = x[1] * w[1] 
xw2 = x[2] * w[2] 
z = xw0 + xw1 + xw2 + b #our output neuron before relu
y = max(z, 0) #relu 

#Now for backward pass
#derivative from the next layer
dvalue = 1.0

#derivative of ReLU and the cahin rule
drelu_dz = dvalue * (1. if z>0 else 0.) #since z > 0 then drelu_dz = 1  drelu/d0 = 0 
print(drelu_dz) 
#derivative of ReLU with respect to z. Lets take the derivatives of parameters of previous layer.
#You can think of it like, If I change z a little bit, how much does output a change? 

#If f(x,y) = x + y. The partials to both of these is 1.
#When we take dsum_dxw0 our partial is the sum with respect to x. For the 0th pair,
#1.0 and -3.0, the sum is 1 we take weighted sum of whole layer for neuron
#. We the multiply with our ReLU function. 
# derivative of xw0 = 1 with respect to the weighted sum z. 
dsum_dxw0 = 1 
drelu_dxw0 = drelu_dz * dsum_dxw0 #our chain rule here. 
dsum_dxw1 = 1
drelu_dxw1 = drelu_dz * dsum_dxw1
dsum_dxw2 = 1
drelu_dxw2 = drelu_dz * dsum_dxw2
dsum_db = 1
drelu_db = drelu_dz * dsum_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db) 

#We know that f(x,y) = x * y  d/dx f(x,y) = 1*y  d/dy f(x,y) = x
#This means when we take derivative of weighted input with respect to input x, we 
#get the weight back. 
#cool tech from the tibeten monks, so drelu_dxw0 is really 
# dReLU / doutputN * dweightedpair
#So the equation below includes the whole chain rule step. 
dmul_dx0 = w[0]
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2
print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

#Lets now expand this whole thing out
drelu_dx0 = drelu_dxw0 * dmul_dx0 
drelu_dx0 = drelu_dxw0 * w[0]
drelu_dxw0 = drelu_dz * dsum_dxw0 
#|------------------------,
drelu_dx0 = drelu_dz * dsum_dxw0 * w[0] 
drelu_dx0 = drelu_dz * 1 * w[0] #the derivative of output wrt weighted pair = 1 always
drelu_dx0 = drelu_dz * w[0]

drelu_dz = dvalue * (1. if z > 0 else 0.) 
drelu_dx0 = dvalue * (1. if z > 0 else 0.) * w[0]

dx = [drelu_dx0, drelu_dx1, drelu_dx2]
dw = [drelu_dw0, drelu_dw1, drelu_dw2]
db = drelu_db 

w[0] += -.001 * dw[0]
w[1] += -.001 * dw[1]
w[2] += -.001 * dw[2] 