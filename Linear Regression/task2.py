import pandas
import os
import matplotlib.pyplot as plt
import numpy as np


def J(x, y, theta):
    return ((x * theta - y).T * (x * theta - y))[0, 0] / (2 * len(x))


def gradient(x, y, theta, alpha, iters):
    costs = list()
    for i in range(0, iters):
        costs.append(J(x, y, theta))
        theta = theta - (alpha / len(x)) * x.T * (x * theta - y)

    return theta, costs


data = pandas.read_csv(os.getcwd() + '/ex1data2.txt', header=None, names=['Square', 'Rooms', 'Price'])
data = (data - data.mean()) / data.std()
data.head()
data.describe()

data.insert(0, 'Ones', 1)
cols = data.shape[1]
x = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.array([[0], [0], [0]])
print("Целевая функция в начале обучения:", J(x, y, theta))
theta1, costs = gradient(x, y, theta, 0.05, 1000)
print("Целевая функция после обучения:", J(x, y, theta1))
print('Theta0 =', theta1[0, 0], "\nTheta1=", theta1[1, 0])
epochs = np.linspace(0, 1000, 1000)
theta2, costs2 = gradient(x, y, theta, 0.01, 1000)
theta3, costs3 = gradient(x, y, theta, 0.1, 1000)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, costs, 'r', label='alpha=0.05')
ax.plot(epochs, costs2, 'g', label='alpha=0.01')
ax.plot(epochs, costs3, 'b', label='alpha=0.1')

ax.legend(loc=1)
ax.set_xlabel('Epoches')
ax.set_ylabel('Losses')
ax.set_title('Losses')
plt.show()