from statistics import mean
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import random

# matplotlib style
style.use('fivethirtyeight')

# manually create data
#xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
#ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# function to create a random dataset
def create_dataset(size, variance, step=2, correlation=False):
    val = 1
    ys = []
    xs = []
    for i in range(size):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
        xs.append(i)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# function to calculate the slope and y intercept of a dataset
def best_fit_line(xs, ys):
    m = (mean(xs) * mean(ys) - mean(xs*ys)) / (mean(xs)**2 - mean(xs**2))
    b = mean(ys) - m * mean(xs)
    return m, b

# function to calculate squared error
def squared_error(y, yhat):
    return sum((y-yhat)**2)

# function to calculate the coefficient of determination
def determination_coefficient(y, yhat):
    y_mean_line = [mean(y) for elem in y]
    se_regression = squared_error(y, yhat)
    se_mean = squared_error(y, y_mean_line)
    return 1 - (se_regression/se_mean)

xs, ys = create_dataset(40, 10, 2, correlation='pos')

m, b = best_fit_line (xs, ys)

# creating the y hat line
regression_line = np.array([m * x + b for x in xs])
# get coefficient of determination
print(determination_coefficient(ys, regression_line))


plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()