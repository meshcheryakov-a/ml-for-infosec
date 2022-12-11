import numpy as np
import pandas as pd

# 1,5,9

def get_object_size(object):
    return np.abs(object[0] - object[2]), np.abs(object[1] - object[3])


# 1. Посчитать все объекты с размерами больше определенных
def count_objects_bigger_size(objects, length, width):
    counter = 0
    for obj in objects:
        obj_length, obj_width = get_object_size(obj)
        if obj_length > length and obj_width > width:
            counter += 1
    return counter


# 5. Определить кадры, когда в кадре было более N объектов одновременно
def get_frames_with_objects(dataframe, max_frame, N):
    frame_indices = np.arange(1, max_frame, 1)
    objects_count = np.zeros(max_frame)
    for idx in frame_indices:
        objects_count[idx] = np.count_nonzero(np.where(dataframe[:, 5] == idx, 1, 0))
    return np.argwhere(objects_count > N)


def get_object_centre(object):
    return (object[0] + object[2]) // 2, (object[1] + object[3]) // 2


def get_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# 9. Определить объекты, которые за N кадров сдвинулись не более чем на M пикселей
def get_moving_objects(objects, N, M):
    moving_objects = []
    for obj in objects:
        object_id = obj[4]
        if object_id in moving_objects:
            continue
        frames_with_object = objects[objects[:, 4] == object_id]
        if len(frames_with_object) >= N:
            move = np.sum(np.abs(np.array(get_object_centre(frames_with_object[0])) -
                                  np.array(get_object_centre(frames_with_object[N - 1]))))
            if move <= M:
                moving_objects.append(object_id)
    return np.array(moving_objects)


def main():
    df = pd.read_csv("MyProject/trajectories.csv", sep=';')
    df = df.drop(columns=['Unnamed: 0'])
    max_frames = 3600
    df = df[df['frame'] < max_frames]  # берем 2 минуты видео исходя из частоты 30 к/с
    print(df)

    count = count_objects_bigger_size(df.values, 200, 100)
    print(count)

    frames = get_frames_with_objects(df.values, max_frames, 50)
    print(frames)

    moving_objects = get_moving_objects(df.values, 100, 200)
    print(moving_objects)


main()