import pandas as pd
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np


def get_bigrams_frequencies(vocabulary, arr):
    tokens_frequencies_arr = np.sum(arr, axis=0)
    tokens_frequencies_dict = {}
    tokens_frequencies_dict.update((key, tokens_frequencies_arr[vocabulary[key]]) for key in vocabulary.keys())
    return dict(sorted(tokens_frequencies_dict.items(), key=lambda item: item[1], reverse=True))


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
    # также здесь выполняется токенизация, убираются стоп-слова, выполняется нормализация
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), stop_words='english', lowercase=True)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    # print(vectorizer.get_stop_words())

    # удаляем пустые строки (нет вхождений ни одного токена)
    indices_to_remove = np.where(np.sum(X_train, axis=1) == 0)
    X_train = np.delete(X_train, indices_to_remove, axis=0)
    y_train = np.delete(y_train, indices_to_remove, axis=0)

    # наиболее встречающиеся биграммы
    bigrams_frequencies = get_bigrams_frequencies(vectorizer.vocabulary_, X_train)
    print("\nbigram", ":", "frequency")
    for bigram in list(bigrams_frequencies.keys())[:20]:
        print(bigram, ":", bigrams_frequencies[bigram])

    # вычисление tf-idf для обучающей выборки
    idf = get_idf(X_train)
    tf = get_tf(X_train)
    X_train = normalize(np.multiply(tf, idf))

    # вычисялем tf-idf для тестовой выборки
    tf = get_tf(X_test)
    X_test = normalize(np.multiply(tf, idf))

    # используем Байессовский классификатор для определения спама
    classificator = MultinomialNB()
    classificator.fit(X_train, y_train)

    y_pred = classificator.predict(X_test)

    # оцениваем качество классификации с помощью метрики accuracy
    confusion_matrix = get_confusion_matrix(y_test.astype(bool), y_pred.astype(bool))

    print("\nconfusion matrix:\n", confusion_matrix)
    print("\naccuracy:", get_accuracy(confusion_matrix))
    print("\naccuracy (sklearn):", accuracy_score(y_test, y_pred))


main()
