'''Регуляризованная нелинейная логистическая регрессия'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import scipy.optimize as opt


def plotData(x, y, xlabel, ylabel, labelPos, labelNeg, X2, title):
    pos = y == 1
    neg = y == 0
    plt.scatter(x[pos, 0], x[pos, 1], s=30, c='g', label=labelPos)
    plt.scatter(x[neg, 0], x[neg, 1], s=30, c='r', label=labelNeg)

    x_min, x_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    y_min, y_max = X2[:, 2].min() - 1, X2[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = sigmoid(poly.fit_transform(np.c_[xx.ravel(), yy.ravel()]).dot(result[0]))
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, [0.5], linewidths=1, colors='b')
    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(x[:, 0].min(), x[:, 0].max())
    plt.ylim(x[:, 1].min(), x[:, 1].max())
    pst = plt.legend(loc='upper right', frameon=True)
    pst.get_frame().set_edgecolor('k')
    #plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunctionR(theta, X, y, lam):
    eps = 0  # 1e-15
    hThetaX = sigmoid(np.dot(X, theta))
    J = - (np.dot(y, np.log(hThetaX)) + np.dot((1 - y), np.log(1 - hThetaX + eps))
           - 1 / 2 * lam * np.sum(np.square(theta[1:]))) / len(y)
    return J


def gradientFuncR(theta, X, y, lam):
    hThetaX = sigmoid(np.dot(X, theta))
    thetaNoZeroReg = np.insert(theta[1:], 0, 0)
    gradient = (np.dot(X.T, (hThetaX - y)) + lam * thetaNoZeroReg) / len(y)
    return gradient


data = np.loadtxt('ex2data2.txt', delimiter=',')
X, y = data[:, :2], data[:, 2]

poly = PolynomialFeatures(6)
# Создаем новые признаки - многочлены до 6 степени
X2 = poly.fit_transform(X)
theta = np.zeros(X2.shape[1])

# Проверка
lam = 1
J = costFunctionR(theta, X2, y, lam)
gradient = gradientFuncR(theta, X2, y, lam)

# cost = 0.693
print("Cost: %0.3f" % J)

result = opt.fmin_tnc(func=costFunctionR,
                      x0=theta, fprime=gradientFuncR,
                      args=(X2, y, lam))
theta_optimized = result[0]
print("Theta: ", theta_optimized)

k = 1
for i, lam in enumerate([0]):#, 0.5, 1, 100]):
    result = opt.fmin_tnc(func=costFunctionR,
                          x0=theta, fprime=gradientFuncR,
                          args=(X2, y, lam))
    if lam == 0:
        title = 'No regularization (Overfitting) (λ = 0)'
    elif lam == 100:
        title = 'Too much regularization (Underfitting) (λ = 100)'
    else:
        title = 'Training data with decision boundary (λ = ' + str(lam) + ')'
    plt.subplot(1, 1, k)
    plotData(X, y, 'Microchip Test 1', 'Microchip Test 2', 'Accepted', 'Rejected', X2, title)

    k += 1
plt.show()
