import numpy as np
import matplotlib.pyplot as plt
import math


def gauss(x, mu, sigma, p=2*np.pi):
    return (1/(sigma*np.sqrt(p))) * np.exp((x-mu)**2/(-2*sigma**2))


def J(x, y, theta):
    _sum = 0
    for i in range(0, len(x)):
        _sum += (h(x[i], theta) - y[i])**2
    return _sum / (2*len(x))


def h(x, theta):
    return theta[0] + x[0] * theta[1] + x[1] * theta[2]


def gradient(x, y, theta, alpha, iters):
    for i in range(0, iters):
        _sum0 = _sum1 = _sum2 = 0
        for i in range(0, len(x)):
            _sum0 += h(x[i], theta) - y[i]
            _sum1 += (h(x[i], theta) - y[i]) * x[i][0]
            _sum2 += (h(x[i], theta) - y[i]) * x[i][1]
        theta[0] = theta[0] - (alpha / len(x)) * _sum0
        theta[1] = theta[1] - (alpha / len(x)) * _sum1
        theta[2] = theta[2] - (alpha / len(x)) * _sum2
    return theta


x_test = np.load('X_test.npy')
y_test = np.load('Y_test.npy')
x_train = np.load('X_train.npy')
y_train = np.load('Y_train.npy')
_x = np.linspace(-1, 2, 100)
plt.subplot(1, 3, 1)
plt.scatter(x_test, y_test)
plt.plot(_x,gauss(_x, 0.5, np.sqrt(0.1)), c='r')
plt.plot(_x,gauss(_x, 0.45, np.sqrt(0.1), 10), c='g')
plt.title('Подбор целевой функции')

x = np.zeros(len(x_test)*2).reshape(len(x_test), 2)
y = np.zeros(len(x_test)).reshape(len(x_test))
for i, x_cur in enumerate(x_test):
    x[i] = x_cur**2, x_cur
    if y_test[i] > 0:
        y[i] = math.log(y_test[i], math.e)
    else:
        y[i] = math.log(0.001, math.e)

theta = theta_opt = np.zeros(3)
print("Целевая функция в начале обучения:", J(x, y, theta))
J_min = J(x, y, theta)
best_alpha = 0
start = 0.005
for alpha in range(1, 11):
    theta1 = gradient(x, y, theta, start, 1000)
    if J(x, y, theta1) < J_min:
        best_alpha = start
        theta_opt = theta1
    start += 0.005
print("Целевая функция после обучения:", J(x, y, theta_opt))
print(theta_opt)
print(best_alpha)

plt.subplot(1, 3, 2)
_x = np.linspace(-1, 2, 100)
plt.scatter(x_test, y_test)
_y = np.array([np.exp(theta_opt[0] + theta_opt[1]*_x[i]**2+theta_opt[2]*_x[i]) for i in range(0,100)])
plt.plot(_x, _y, c='r')
plt.title('Тестовый набор')

plt.subplot(1, 3, 3)
plt.scatter(x_train, y_train)
plt.plot(_x, _y, c='r')
plt.title('Проверочный набор')

x2 = np.zeros(len(x_train)*2).reshape(len(x_train), 2)
y2 = np.zeros(len(x_train)).reshape(len(x_train))
for i, x_cur in enumerate(x_train):
    x2[i] = x_cur**2, x_cur
    if y_train[i] > 0:
        y2[i] = math.log(y_train[i], math.e)
    else:
        y2[i] = math.log(0.001, math.e)
print("Целевая функция на проверочном наборе:", J(x2, y2, theta_opt))
plt.show()
