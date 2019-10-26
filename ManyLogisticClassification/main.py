import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunction(theta, X, y):
    eps = 0  # 1e-15

    hThetaX = sigmoid(np.dot(X, theta))
    J = - (np.dot(y, np.log(hThetaX)) + np.dot((1 - y), np.log(1 - hThetaX + eps))) / len(y)
    return J


def gradientFunc(theta, X, y):
    hThetaX = sigmoid(np.dot(X, theta))
    gradient = np.dot(X.T, (hThetaX - y)) / len(y)
    return gradient


def h(X, theta):
    res = 0
    for i in range(0, len(X)):
        res += X[i] * theta[i + 1]
    return theta[0] + res


def predict(X, theta):
    return sigmoid(h(X, theta))


data = np.loadtxt('ex3data.txt', delimiter=',')
old_X, y = data[:, :-1], data[:, -1]
X = np.hstack((np.ones((old_X.shape[0], 1)), old_X))

theta = np.zeros(X.shape[1])
theta_res = np.zeros((10, 401))

for i in range(1, 11):
    new_y = np.copy(y)
    new_y[500*(i-1):500 * i] = 1
    new_y[:500*(i-1)] = 0
    new_y[500 * i + 1:] = 0
    result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradientFunc, args=(X, new_y))
    theta_optimized = result[0]
    theta_res[i-1] = theta_optimized
    print(i-1)

right = 0
h = sigmoid(np.dot(X, theta_res.T))
h_argmax = np.argmax(h, axis=1)
for i, x in enumerate(h_argmax):
    print(f'y={y[i]}, predict={x}')
    if int(y[i]) == x:
        right += 1

print(f'Accuracy: {right/5000.0}')
