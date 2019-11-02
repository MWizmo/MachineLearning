import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
print(type(iris))
print(iris.feature_names)
print(iris.target_names)

# %%Plot a simple scatter plot of 2 features of the iris dataset

# The indices of the features that we are plotting
x_index = 0
y_index = 1

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

# plt.figure(figsize=(5, 4))
# plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
# plt.colorbar(ticks=[0, 1, 2], format=formatter)
# plt.xlabel(iris.feature_names[x_index])
# plt.ylabel(iris.feature_names[y_index])
# plt.tight_layout()
#
# x_index = 2
# y_index = 3
# formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
# plt.figure(figsize=(5, 4))
# plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
# plt.colorbar(ticks=[0, 1, 2], format=formatter)
# plt.xlabel(iris.feature_names[x_index])
# plt.ylabel(iris.feature_names[y_index])
#
# plt.tight_layout()
#plt.show()

# %% Входные данные X и правильные ответы y
X = iris.data
y = iris.target


def euclidean_distance(row1, row2):
    _sum = sum([(row1[i]-row2[i])**2 for i in range(0, len(row1))])
    return math.sqrt(_sum)


# Функция для нахождения ближайших соседей
def get_neighbors(train_set, labels, test_row, num_neighbors):
    distances = list()
    for index in range(len(train_set)):
        dist = euclidean_distance(test_row, train_set[index])
        if dist != 0.0:
            distances.append((train_set[index], dist, labels[index]))
    distances.sort(key=lambda tup: tup[1])
    neighbors = distances[:num_neighbors]
    return neighbors


# Функция для предсказания класса объекта
def predict_classification(train_set, labels, test_row, num_neighbors):
    classes = get_neighbors(train_set, labels, test_row, num_neighbors)
    votes = [0, 0, 0]
    for c in classes:
        votes[c[2]] += 1/c[1]
    return votes.index(max(votes))


# kNN метод
def k_nearest_neighbors(train_set, labels, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train_set, labels, row, num_neighbors)
        predictions.append(output)
    return predictions


# %% Тестируем функции
dataset = X[:150:15]
output = y[:150:15]
for i in range(0, len(dataset)):
    print(euclidean_distance(dataset[i], dataset[5]))

neighbors = get_neighbors(dataset, output, dataset[5], 3)
for neighbor in neighbors:
    print(neighbor)


nums = list()
accuracies = list()
for k in range(1, 61):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    res = k_nearest_neighbors(X_train, y_train, X_test, k)
    right = 0
    for i in range(0, len(res)):
        # print('Expected %d, Got %d.' % (y_test[i], res[i]))
        if y_test[i] == res[i]:
            right += 1
    #print(f"Accuracy: {right/len(res)}")
    nums.append(k)
    accuracies.append((right/len(res))*100)

plt.plot(nums, accuracies)
plt.xlabel('Количество соседей')
plt.ylabel('Точность классификатора')
plt.show()

# %% Встроенный в sklearn метод kNN
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
num_neighbors = 5
knn = KNeighborsClassifier(num_neighbors)  # , weights='distance')
knn.fit(X_train, y_train)
predictions_sk = knn.predict(X_test)
right = 0
for i in range(0, len(predictions_sk)):
    if y_test[i] == predictions_sk[i]:
        right += 1
print(f"Accuracy1: {right/len(predictions_sk)}")

# %% Встроенный в sklearn метод логистической регрессии
from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
predictions_lr = LogReg.predict(X_test)
right = 0
for i in range(0, len(predictions_lr)):
    if y_test[i] == predictions_lr[i]:
        right += 1
print(f"Accuracy2: {right/len(predictions_lr)}")
