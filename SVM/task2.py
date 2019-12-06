import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import mglearn

data = loadmat('ex5data2.mat')
X = data["X"]
y = data["y"].ravel()

svmclassifier = svm.SVC(kernel='rbf', C=1000, gamma=10)
svmclassifier.fit(X, y)
print('Accuracy:', svmclassifier.score(X, y))
mglearn.plots.plot_2d_separator(svmclassifier, X, eps=.05)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.show()