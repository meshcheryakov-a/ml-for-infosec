import pandas as pd
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np


def get_tf(arr):
    # res = np.ndarray(shape=arr.shape)
    # for i in range(arr.shape[0]):
    #     sum = np.sum(arr[i])
    #     res[i] = arr[i] / (sum if sum != 0 else 1)
    # return res
    return arr


def get_idf(arr):
    N = arr.shape[0]
    res = np.copy(arr)
    res[arr > 0] = 1
    df = np.sum(res, axis=0)
    return np.log((N + 1) / (df + 1)) + 1


def get_confusion_matrix(y_true, y_pred):
    tp = np.sum(y_pred & y_true)
    fp = np.sum(y_pred[y_true == False])
    tn = np.sum(~y_pred & ~y_true)
    fn = np.sum(~y_pred[y_true == True])
    return np.array([[tp, fp], [fn, tn]])


def get_accuracy(confusion_matrix):
    return np.trace(confusion_matrix) / np.sum(confusion_matrix.flatten())


def main():
    # чтение данных
    df = pd.read_csv("data/spam.csv", usecols=[0, 1], encoding='latin-1')
    # print(df)

    # кодируем строки в числа (ham => 0, spam => 1)
    le = LabelEncoder()
    df['v1'] = le.fit_transform(df['v1'])

    print(df)
    # разбиваем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(df['v2'].values, df['v1'].values, test_size=0.33,
                                                        random_state=42)

    # преобразование текста в вектор признаков (Bag of Words)
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), stop_words='english', lowercase=True)
    X_train = vectorizer.fit_transform(X_train).todense()
    X_test = vectorizer.transform(X_test).todense()

    print(vectorizer.vocabulary_)

    tokens = vectorizer.vocabulary_.keys()
    print(len(tokens))

    idf = get_idf(X_train)
    tf = get_tf(X_train)
    # print(tf.shape)
    # print(idf.shape)

    res_train = np.multiply(tf, idf)
    res_train = normalize(res_train)

    # tfidf = TfidfTransformer(norm=None)
    # X_train = tfidf.fit_transform(X_train).todense()
    # X_test = tfidf.transform(X_test).todense()

    classificator = MultinomialNB()
    classificator.fit(X_train, y_train)

    tf = get_tf(X_test)
    res_test = np.multiply(tf, idf)
    X_test = normalize(res_test)

    y_pred = classificator.predict(X_test)
    confusion_matrix = get_confusion_matrix(y_test.astype(bool), y_pred.astype(bool))
    print(get_accuracy(confusion_matrix))


main()
