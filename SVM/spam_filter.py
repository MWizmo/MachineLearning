import re
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn import svm
from scipy.io import loadmat
from sklearn.externals import joblib


def processEmail(email_contents):
    word_indices = []
    email_contents = email_contents.lower()
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    email_contents = email_contents.translate(str.maketrans('', '', punctuation))
    email_contents = email_contents.split()
    stemmer = SnowballStemmer("english")
    for token in email_contents:
        token = stemmer.stem(token.strip())
        if token in vocabList:
            idx = vocabList[token]
            word_indices.append(idx)
    return word_indices


def emailFeatures(word_indices):
    n = 1899
    x = np.zeros((n, 1), dtype='int64')
    for idx in word_indices:
        x[idx] = 1
    return x


def predict_email(model, file):
    email = open(file, 'r').read()
    indices = processEmail(email)
    x = emailFeatures(indices).reshape(1, 1899)
    res = model.predict(x)
    if res[0] == 0:
        return 'not spam'
    else:
        return 'spam'


f = open('emailSample1.txt', 'r').read()
print(f)
vocab = open('vocab.txt', 'r')
vocabList = {}
for line in vocab.readlines():
    i, word = line.split()
    vocabList[word] = int(i)
vocab.close()
word_indices = processEmail(f)
print('Word Indices:')
print(word_indices)
features = np.array(emailFeatures(word_indices).flatten())
print('Length of feature vector: {:d}'.format(len(features)))
print('Number of non-zero entries: {:d}'.format(np.sum(features > 0)))

data1 = loadmat('spamTrain.mat')
X_train = data1["X"]
y_train = data1["y"].ravel()
data2 = loadmat('spamTest.mat')
X_test = data2["Xtest"]
y_test = data2["ytest"].ravel()
svc = svm.LinearSVC()
svc.fit(X_train, y_train)
joblib.dump(svc, 'model.pkl')
# svc = joblib.load('model.pkl')
print("Accuracy on train set:", svc.score(X_train, y_train))
print("Accuracy on test set:", svc.score(X_test, y_test))
print(predict_email(svc, 'emailSample1.txt'))
print(predict_email(svc, 'emailSample2.txt'))
print(predict_email(svc, 'spamSample1.txt'))
print(predict_email(svc, 'spamSample2.txt'))
