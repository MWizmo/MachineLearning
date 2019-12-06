# Строим классификатор спама
import re
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn import svm
from scipy.io import loadmat

# %% ======== Предобработка и перевод письма в список индексов слов словаря ============

# Посмотрим на письмо
f = open('emailSample1.txt', 'r').read()
print(f)

# %% Считаем словарь
vocab = open('vocab.txt', 'r')
vocabList = {}
for line in vocab.readlines():
    i, word = line.split()
    vocabList[word] = int(i)
vocab.close()


# %% Функция для предобработки данных
def processEmail(email_contents):
    """
    Функция предобработки данных и преобразования письма
    в список индексов слов словаря
    Вход:
        email_contents: str
    Выход:
        word_indices: list
    """
    # Это будет письмо, преобразованное в индексы слов словаря
    word_indices = []

    # ============================ Предобработка ============================

    # Cтрочный регистр
    email_contents = email_contents.lower()

    # Уберем HTML
    # Ищем все выражения, которые начинаются с < и оканчиваются на > и не
    # содержат  < или > в тегах и заменяем на пробел
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Все цифры заменим на number
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Нормализация URLS
    # Ищем строки, начинающие на http:// или https://, заменяем на 'httpaddr'
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Нормализация Email
    # Ищем строки с @ в середине
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Замена $ на "dollar".
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    # Избавимся от знаков препинания
    email_contents = email_contents.translate(str.maketrans('', '', punctuation))

    # ============================ Tokenize Email ============================

    # Split текст
    email_contents = email_contents.split()

    # Создаем стеммер
    stemmer = SnowballStemmer("english")

    # Выведем обработанное письмо
    print('Processed Email')

    for token in email_contents:

        # Стемминг слов
        token = stemmer.stem(token.strip())

        # Заполняем word_indices
        if token in vocabList:
            idx = vocabList[token]
            word_indices.append(idx)

        # ====================================================================

        print(token, end=' ')

    print('\n\n=========================\n')

    return word_indices


# %% Выведем обработанное письмо и его список индексов слов словаря

word_indices = processEmail(f)
print('Word Indices: \n')
print(word_indices)
print('\n\n=========================\n')


# %%  Функция для создания вектора признаков объекта-письма
def emailFeatures(word_indices):
    """
    Функция создает вектор
    признаков данного письма
    из списка word_indices
    Вход:
        word_indices: list
    Выходs:
        x: бинарный вектор признаков (n, 1)
    """
    # Сколько всего слов в словаре
    n = 1899

    # Инициализация
    x = np.zeros((n, 1))

    # Присваиваем x[idx] 1, если idx есть в word_indices
    for idx in word_indices:
        x[idx] = 1

    return x


# %% Преобразуем word_indices в вектор признаков

features = emailFeatures(word_indices)

# Выведем число ненулевых элементов
print('Length of feature vector: {:d}'.format(len(features)))
print('Number of non-zero entries: {:d}'.format(np.sum(features > 0)))

# %% ============ Использование SVM ============================
