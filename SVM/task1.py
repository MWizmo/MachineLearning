import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import mglearn

data = loadmat('ex5data1.mat')
X = data["X"]
y = data["y"].ravel()
c = [1, 100, 1000]
svc = list()
for _c in c:
    svc.append(svm.LinearSVC(C=_c, loss='hinge', max_iter=10000))
for i in range(len(c)):
    svc[i].fit(X, y)
    print('C =', c[i], ' Accuracy:', svc[i].score(X, y))
    mglearn.plots.plot_2d_separator(svc[i], X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.show()
