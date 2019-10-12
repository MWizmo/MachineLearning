import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def plotData(x, y, xlabel, ylabel, labelPos, labelNeg, x_border, y_border):
    pos = y == 1
    neg = y == 0
    plt.scatter(x[pos, 0], x[pos, 1], s=30, c='green', marker='+', label=labelPos)
    plt.scatter(x[neg, 0], x[neg, 1], s=30, c='red', label=labelNeg)
    plt.plot(x_border, y_border, 'b', label='Border')
    plt.xlim(x[:, 0].min(), x[:, 0].max())
    plt.ylim(x[:, 1].min(), x[:, 1].max())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    pst = plt.legend(loc='upper right', frameon=True)
    pst.get_frame().set_edgecolor('k')
    plt.show()


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


data = np.loadtxt('ex2data1.txt', delimiter=',')
old_X, y = data[:, :2], data[:, 2]
X = np.hstack((np.ones((old_X.shape[0], 1)), old_X))

theta = np.zeros(X.shape[1])

J = costFunction(theta, X, y)
gradient = gradientFunc(theta, X, y)

print("Cost: %0.3f".format(J))
print("Gradient: {0}".format(gradient))

result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradientFunc, args=(X, y))
theta_optimized = result[0]
print(theta_optimized)

x_border = np.linspace(0, 100, 100)
y_border = (-theta_optimized[0] - (theta_optimized[1] * x_border)) / theta_optimized[2]

plotData(old_X, y, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not Admitted', x_border, y_border)
print("Вероятность поступить: ", predict([45, 85], theta_optimized))
right = 0
for row in data:
    if predict([row[0], row[1]], theta_optimized) > 0.5 and row[2] == 1.0:
        right += 1
    elif predict([row[0], row[1]], theta_optimized) < 0.5 and row[2] == 0.0:
        right += 1

print('Точность классификатора:', right / len(data))
