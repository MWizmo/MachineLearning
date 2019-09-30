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


data = pandas.read_csv(os.getcwd() + '/ex1data1.txt', header=None, names=['Population', 'Profit'])
data.head()
data.describe()

data.insert(0, 'Ones', 1)
cols = data.shape[1]
x = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.array([[0], [0]])
print("Целевая функция в начале обучения:", J(x, y, theta))
theta1, costs1 = gradient(x, y, theta, 0.01, 1000)
theta2, costs2 = gradient(x, y, theta, 0.005, 1000)
theta3, costs3 = gradient(x, y, theta, 0.001, 1000)
theta4, costs4 = gradient(x, y, theta, 0.02, 1000)
epochs = np.linspace(0, 1000, 1000)

x_predict = np.linspace(data.Population.min(), data.Population.max(), 100)
y_predict1 = theta1[0, 0] + (theta1[1, 0] * x_predict)
y_predict2 = theta2[0, 0] + (theta2[1, 0] * x_predict)
y_predict3 = theta3[0, 0] + (theta3[1, 0] * x_predict)
y_predict4 = theta4[0, 0] + (theta4[1, 0] * x_predict)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_predict, y_predict1, 'r', label='alpha=0.01, J(x,y)=' + str(J(x, y, theta1)))
ax.plot(x_predict, y_predict2, 'g', label='alpha=0.005, J(x,y)=' + str(J(x, y, theta2)))
ax.plot(x_predict, y_predict3, 'b', label='alpha=0.001, J(x,y)=' + str(J(x, y, theta3)))
ax.plot(x_predict, y_predict4, 'y', label='alpha=0.02, J(x,y)=' + str(J(x, y, theta4)))
ax.scatter(data.Population, data.Profit, label='Training data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Prediction')

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(epochs, costs1, 'r', label='alpha=0.01')
ax2.plot(epochs, costs2, 'g', label='alpha=0.005')
ax2.plot(epochs, costs3, 'b', label='alpha=0.001')
ax2.plot(epochs, costs4, 'y', label='alpha=0.02')
ax2.legend(loc=1)
ax2.set_xlabel('Epoches')
ax2.set_ylabel('Losses')
ax2.set_title('Losses')

plt.show()