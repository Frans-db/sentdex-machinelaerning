from math import sqrt
from matplotlib import style
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import warnings

style.use('fivethirtyeight')

dataset = {'k': [[1,2], [2,3], [3,1 ]], 'r': [[6,5],[7,7],[8,6]]}
new_features = [5,7]

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)
plt.show()

def k_nearest_neighbours(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than the total amount of voting groups')
    
    