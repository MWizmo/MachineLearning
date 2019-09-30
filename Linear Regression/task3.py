from sklearn import linear_model
import pandas
import numpy as np
import matplotlib.pyplot as plt
import os

data = pandas.read_csv(os.getcwd() + '/ex1data1.txt', header=None, names=['Population', 'Profit'])
data.head()
data.describe()

data.insert(0, 'Ones', 1)
cols = data.shape[1]
x = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

x = np.matrix(x.values)
y = np.matrix(y.values)

model = linear_model.LinearRegression()
model.fit(x, y)
f1 = model.predict(x)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.array(x[:, 1]), f1, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Prediction')
plt.show()