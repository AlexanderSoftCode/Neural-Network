import random 
import numpy  as np
#Classes.py
from Classes import * 

dropout_rate = 0.3
example_output = np.array([0.27, -1.03, 0.67, 0.99, 0.05,
                           -0.37, -2.01, 1.13, -0.07, 0.73])
print(f'sum initial {sum(example_output)}')
sums = []

for i in range(10000): 
    example_output2 = example_output * np.random.binomial(1, 1-dropout_rate, example_output.shape) \
                    / (1-dropout_rate)
    sums.append(sum(example_output2))

print(f'mean sum: {np.mean(sums)}')

#np.random.binomial(​2​, ​ 0.5​, ​ size​=​10​)
#array([​0​, ​ 0​, ​ 1​, ​ 2​, ​ 0​, ​ 2​, ​ 0​, ​ 1​, ​ 0​, ​ 2​]) 
#This means we have 2 coin tosses per sample. Each coin toss has .5 chance of landing heads or 
#tails so that's how it works. 