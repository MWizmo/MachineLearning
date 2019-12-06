import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import mglearn

data = loadmat('ex5data3.mat')
X = data["X"]
y = data["y"].ravel()
Xval = data["Xval"]
yval = data["yval"].ravel()

max_acc = 0
best_svm = None
params = []

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

for c in C_values:
    for gamma in gamma_values:
        svmclassifier = svm.SVC(kernel='rbf', C=c, gamma=gamma)
        svmclassifier.fit(X, y)
        acc = svmclassifier.score(Xval, yval)
        if acc > max_acc:
            max_acc = acc
            best_svm = svmclassifier
            params = [c, gamma]

print('Best accuracy:', max_acc)
print('Best params:', params)
mglearn.plots.plot_2d_separator(best_svm, Xval, eps=.05)
mglearn.discrete_scatter(Xval[:, 0], Xval[:, 1], yval)
plt.show()