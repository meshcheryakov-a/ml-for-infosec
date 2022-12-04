import numpy as np


def filter_car_bboxes(car_bboxes):
    new_car_bboxes = []
    delta = 20
    coord_1 = car_bboxes[0, 0]
    coord_2 = car_bboxes[0, 1]
    coord_3 = car_bboxes[0, 2]
    coord_4 = car_bboxes[0, 3]
    for box in car_bboxes[1:]:
        if not (np.abs(coord_1 - box[0]) < delta or np.abs(coord_2 - box[1]) < delta \
            or np.abs(coord_3 - box[2]) < delta or np.abs(coord_4 - box[3]) < delta):
            new_car_bboxes.append(box)
            coord_1 = box[0]
            coord_2 = box[1]
            coord_3 = box[2]
            coord_4 = box[3]
    return np.array(new_car_bboxes)


# взять что-то, кроме 2,3,6

