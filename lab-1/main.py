import pandas as pd
import numpy as np

# реализация метода, который не поднимет исключение KeyError, а вернет 0, если нет заданного ключа в словаре
class my_dict(dict):
    def __missing__(self, key):
        return 0


def get_next_state(current_state, elem):
    next_state = []
    # если размер окна не равен 1, то делаем сдвиг элементов влево
    if current_state.size != 1:
        next_state = current_state[1:]
    next_state = np.append(next_state, int(elem))
    return next_state.astype(int)


def get_matrix_from_dict(states, transition_dict):
    # матрица вероятностей перехода между состояниями
    transition_matrix = [list(states)]
    transition_matrix[0].insert(0, len(states))

    # формируем матрицу переходов размера NxN, N = количество состояний
    for state in states:
        # вектор вероятностей перехода из состояния state в другие состояния
        transition_vector = [state]

        for next_state in states:
            # ищем вероятность перехода из состояния state в next_state
            transition = (state, next_state)
            probability = transition_dict[transition]
            transition_vector.append(probability)

        transition_matrix.append(transition_vector)

    return transition_matrix


def get_transition_matrix(elements, window_size):
    # словарь вероятностей переходов
    transition_dict = my_dict()

    # количество переходов из каждого состояния, необходимо для подсчета вероятности
    transitions_count = my_dict()

    # начальное состояние
    current_state = np.ones(window_size, dtype=int)

    for elem in elements:
        next_state = get_next_state(current_state, elem)

        # переводим np.array в tuple для создания ключа в словаре
        current_state_tuple = tuple(current_state.tolist())
        next_state_tuple = tuple(next_state.tolist())

        transitions_count[current_state_tuple] += 1

        transition = (current_state_tuple, next_state_tuple)
        transition_dict[transition] += 1

        current_state = next_state

    # считаем вероятности переходов между состояниями
    for key in transition_dict.keys():
        transition_dict[key] /= transitions_count[key[0]]

    states = transitions_count.keys()

    transition_matrix = get_matrix_from_dict(states, transition_dict)

    return transition_matrix


def check_transition_matrix(transition_matrix):
    # вероятности перехода из каждого состояния, должны быть равны 1
    probabilities = []

    transition_matrix = transition_matrix[1:]
    for row in transition_matrix:
        probabilities.append(np.sum(row[1:]))

    if (probabilities != np.ones(len(probabilities), dtype=float)).all():
        return False
    return True


def predict_anomalies(elements, transition_matrix, window_size, threshold):
    # получаем состояния из матрицы переходов
    states = transition_matrix[0][1:]

    # 1 - аномальное значение, 0 - нормальное значение
    anomalies = []

    # начальное состояние
    current_state = np.ones(window_size, dtype=int)

    for elem in elements:
        next_state = get_next_state(current_state, elem)

        # переводим np.array в tuple
        current_state_tuple = tuple(current_state.tolist())
        next_state_tuple = tuple(next_state.tolist())

        try:
            # получаем индексы состояний в матрице переходов
            i = states.index(current_state_tuple) + 1
            j = states.index(next_state_tuple) + 1

            # вероятность перехода по матрице переходов
            transition_probability = transition_matrix[i][j]

            # обнаружение аномалии, если вероятность перехода ниже порогового значения
            if transition_probability < threshold:
                anomalies.append(1)
            else:
                anomalies.append(0)

        except ValueError:  # если не найдено состояние в матрице переходов, то это считается аномалией
            anomalies.append(1)

        current_state = next_state

    return anomalies


def get_anomalies_data(df, transition_matrices, window_size, threshold):
    # определяем аномалии в массиве данных data fake
    anomalies_data = []

    for (value, transition_matrix) in zip(df['value'], transition_matrices):
        # переводим данные из строки в массив
        elements = np.array(value.split(';')).astype(int)

        anomalies_data.append(predict_anomalies(elements, transition_matrix, window_size, threshold))

    return anomalies_data


def main():
    # чтение данных
    df = pd.read_csv("data/data.txt", sep=':')
    df_true = pd.read_csv("data/data_true.txt", sep=':')
    df_fake = pd.read_csv("data/data_fake.txt", sep=':')
    # сортировка данных в data_fake.txt по полю user
    df_fake = df_fake.sort_values(by='users', key=lambda col: col.str.removeprefix('user').astype(int))

    # матрицы переходов для каждого пользователя
    transition_matrices = []

    window_size = int(input('Enter window size: '))

    # составляем матрицы переходов для каждого пользователя
    for value in df['value']:
        # переводим данные из строки в массив
        elements = np.array(value.split(';')).astype(int)

        transition_matrix = get_transition_matrix(elements, window_size)

        # проверка матрицы перехода на соответствие условию: сумма вероятностей по строкам должна быть равна 1
        status = check_transition_matrix(transition_matrix)
        if status == True:
            transition_matrices.append(transition_matrix)
        else:
            print("Transition matrix is invalid")
            exit(1)

    # пороговое значение вероятности перехода из одного состояния в другое для обнаружения аномалий
    threshold = float(input('Enter threshold probability: '))

    # определяем аномалии в массиве данных data true
    anomalies_data_true = get_anomalies_data(df_true, transition_matrices, window_size, threshold)

    # определяем аномалии в массиве данных data fake
    anomalies_data_fake = get_anomalies_data(df_fake, transition_matrices, window_size, threshold)

    print(f'Model params:\nwindow_size = {window_size}\nthreshold probability = {threshold}')
    print(f'Detected {np.sum(anomalies_data_true)} anomalies in data_true.txt.')
    print(f'Detected {np.sum(anomalies_data_fake)} anomalies in data_fake.txt.')

    answer = input('Get transition matrix for user: ')
    user_index = df.loc[df['users'] == answer].index.values[0]
    for row in transition_matrices[user_index]:
        print(row)


main()