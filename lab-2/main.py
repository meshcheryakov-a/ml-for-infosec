import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
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

    # преобразование текста в вектор признаков
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), stop_words='english')
    X_train = vectorizer.fit_transform(X_train).todense()

    tokens = vectorizer.vocabulary_.keys()
    print(len(tokens))

    idf = get_idf(X_train)
    tf = get_tf(X_train)
    print(tf.shape)
    print(idf.shape)

    print(type(idf))

    res = np.multiply(tf, idf)

    tfidf = TfidfTransformer(norm=None)
    X_train = tfidf.fit_transform(X_train).todense()

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if res[i, j] != X_train[i, j]:
                print(tf[i, j] * idf[j])
                print(res[i, j], X_train[i, j], i, j, tf[i, j], idf[j], tfidf.idf_[j])



main()