import os
import time

import imageio
import numpy as np
import pandas as pd
from skimage import io
from sort_tracker import SortTracker
from configuration import BaseConfig
import cv2
from utils import filter_car_bboxes

class ObjectTracker:
    def __init__(self, **kwargs):
        start_time = time.time()
        _FILE_NAME = kwargs.get('FILE_NAME')
        _PROJECT_NAME = kwargs.get('PROJECT_NAME')
        _PROJECT_DIR = os.path.join(BaseConfig._base_dir, _PROJECT_NAME)
        _DETECTS_DIR = os.path.join(_PROJECT_DIR, 'detects')
        _CNNType = kwargs.get('CNNType')
        _BORDER_CROP = kwargs.get('BORDER_CROP')

        for path in [_PROJECT_DIR, _DETECTS_DIR]:
            if not os.path.exists(path):
                os.makedirs(path)

        trajectories = pd.DataFrame(columns=['x_right', 'y_bottom', 'x_left', 'y_top', 'ID', 'frame'])
        video = imageio.get_reader(os.path.join(_PROJECT_DIR, 'video', _FILE_NAME), 'ffmpeg')
        detector = ObjectsCNNRecognizer_CV2(**kwargs)
        tracker = SortTracker()
        counter = 0

        # цикл по кадрам видео
        for index, frame in enumerate(video):
            read_start_time = time.time()
            counter += 1
            print(frame.shape)
            new_frame = frame.copy()
            detector_start_time = time.time()
            # обнаружение объектов в новом кадре
            car_bboxes, car_masks = detector.get_recognized_objects(new_frame)
            detector_finish_time = time.time() - detector_start_time
            print("--- Время работы детектора, затраченное на обнаружение объектов --- {time:.3f} seconds --- \n".format(time=detector_finish_time))

            # если в кадре не нашлось машин
            if car_bboxes.shape[0] == 0:
                continue
            else:

                # убираем повторяющиеся объекты
                car_bboxes = filter_car_bboxes(car_bboxes)

                tracker_start_time = time.time()
                tracked_array = np.round(tracker.update(np.column_stack([car_bboxes[:, 1], car_bboxes[:, 0], car_bboxes[:, 3], car_bboxes[:, 2]]))).astype(int)
                print(car_bboxes)
                tracked_df = pd.DataFrame(columns=['y_top', 'x_left', 'y_bottom', 'x_right', 'ID'])
                tracked_df['x_left'] = pd.Series(tracked_array[:, 0]).apply(lambda x: max(x - _BORDER_CROP, 0))
                tracked_df['x_right'] = pd.Series(tracked_array[:, 2]).apply(lambda x: min(x + _BORDER_CROP, frame.shape[1]))
                tracked_df['y_bottom'] = pd.Series(tracked_array[:, 3]).apply(lambda x: min(x + _BORDER_CROP, frame.shape[0]))
                tracked_df['y_top'] = pd.Series(tracked_array[:, 1]).apply(lambda x: max(x - _BORDER_CROP, 0))
                tracked_df['ID'] = pd.Series(tracked_array[:, 4])
                tracked_df['frame'] = counter
                trajectories = trajectories.append(tracked_df)
                trajectories = trajectories.reset_index(drop=True)
                detector_start_time = time.time()
                for car_id in range(len(tracked_array)):
                    x_left = max(tracked_array[car_id, 0] - _BORDER_CROP, 0)
                    x_right = min(tracked_array[car_id, 2] + _BORDER_CROP, frame.shape[1])
                    y_bottom = min(tracked_array[car_id, 3] + _BORDER_CROP, frame.shape[0])
                    y_top = max(tracked_array[car_id, 1] - _BORDER_CROP, 0)
                    print(x_left, x_right, y_bottom, y_top, tracked_array[car_id, 4])

                    # выделяем объекты на кадре
                    cv2.rectangle(new_frame, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 3)

                    distance_temp = (np.array(np.abs(car_bboxes[:, 2] - tracked_array[car_id, 3]) +
                                              np.abs(car_bboxes[:, 3] - tracked_array[car_id, 2]) +
                                              np.abs(car_bboxes[:, 0] - tracked_array[car_id, 1]) +
                                              np.abs(car_bboxes[:, 1] - tracked_array[car_id, 0])))
                    d_img_to_draw = (frame[y_top:y_bottom:, x_left:x_right, :])
                    if (d_img_to_draw.shape[0] > 0) and (d_img_to_draw.shape[1] > 0):
                        io.imsave(os.path.join(_DETECTS_DIR, 'car_id_{}_{}.png'.format(tracked_array[car_id, 4], counter)), d_img_to_draw / d_img_to_draw.max())

                # выводим кадр с помеченными объектами
                cv2.namedWindow("output", cv2.WINDOW_NORMAL)
                cv2.imshow("output", new_frame)
                cv2.waitKey(500)

            if counter % 10 == 0:
                trajectories.to_csv(os.path.join(_PROJECT_DIR, 'trajectories.csv'), sep=';')
            print("--- Затрачено на кадр № {counter} --- {time:.3f} seconds --- \n".format(counter=counter, time=time.time() - read_start_time))

        print("--- {last_frame} кадров из видео {video_name} отбрабатывалось {time:.3f} seconds --- \n".format(last_frame=counter,
                                                                                                               video_name=_FILE_NAME,
																											   time=time.time() - start_time))
class ObjectsCNNRecognizer_CV2():
    """ Класс, представляющий собой нейросетевой детектор на базе MobileNetSSD
    Метод __init__ подгружает caffe-модель и создает экземпляр детектора
    Метод get_recognized_objects возвращает найденные объекты на переданном ему изображении
    """
    def __init__(self, **kwargs):
        self.CNNType = kwargs.get('CNNType')
        confidence = kwargs.get('DETECTION_MIN_CONFIDENCE')
        if self.CNNType=='MobileNET':
            self.prototxt = BaseConfig._base_dir + '/ObjectsCNNRecognizer_CV2/MobileNetSSD_deploy.prototxt.txt'
            self.model = BaseConfig._base_dir + '/ObjectsCNNRecognizer_CV2/MobileNetSSD_deploy.caffemodel'
            if confidence=='':
                self.confidence = 0.33
            else:
                self.confidence = float(confidence)
            # initialize the list of class labels MobileNet SSD was trained to
            # detect, then generate a set of bounding box colors for each class
            self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                            "sofa", "train", "tvmonitor"]
            self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
            # load our serialized model from disk
            print("[INFO] loading model...")
            self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        elif self.CNNType=='YOLO':
            if confidence=='':
                self.confidence = 0.33
            else:
                self.confidence = float(confidence)
            self.prototxt = BaseConfig._base_dir + '/ObjectsCNNRecognizer_CV2/yolo.cfg'
            self.model = BaseConfig._base_dir + '/ObjectsCNNRecognizer_CV2/yolov3.weights'
            self.net = cv2.dnn.readNetFromDarknet(self.prototxt,self.model)
        else:
            print ('Unknown CNN type!')

    def get_recognized_objects(self, frame):
        """ Метод, возвращающий найденные объекты на кадре
        :param frame:
            Входное изображение, на котором производится поиск объектов
        """
        (frameHeight, frameWidth) = frame.shape[:2]
        #blob = cv2.dnn.blobFromImage(frame, scalefactor=0.007843, size=(300, 300), mean=(127,127,127))
        boxes=np.zeros([0,4]) # Массив с координатами найденных машин, строка =машина
        if self.CNNType=='YOLO':
            self.net.setInput(cv2.dnn.blobFromImage(frame, scalefactor=0.00392,size=(416, 416), mean=(0,0,0),swapRB=False,crop=False))
            outs = self.net.forward()     
            for detection in outs:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if (confidence > self.confidence) and (int(classId)==2):
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    bottom = int(center_y + height / 2)
                    right= int(center_x + width / 2)
                    boxes=np.vstack([boxes,[top,left, bottom,right]])
                    boxes=boxes.astype(int)
        elif self.CNNType=='MobileNET':
            self.net.setInput(cv2.dnn.blobFromImage(cv2.resize(frame, (1080, 720)), scalefactor=0.00784 ,size=(1080, 720), mean=(127.5,127.5,127.5),swapRB=True,crop=False))
            outs = self.net.forward()     
            for out in outs:
                    for detection in outs[0, 0,:,:]:
                        confidence = detection[2]
                        object_type= detection[1]
                        if (int(object_type)==7) and (confidence > self.confidence):
                            x_left = int(detection[3] * frameWidth)
                            y_top = int(detection[4] * frameHeight)
                            x_right = int(detection[5] * frameWidth)
                            y_bottom = int(detection[6] * frameHeight)
                            boxes=np.vstack([boxes,[y_top,x_left, y_bottom,x_right ]])
                            boxes=boxes.astype(int)   
        else:
            print('Unknown CNN model!')
        return boxes, None
        
parameters = dict(
    # Параметры запуска трекера
    FILE_NAME='TrafficHD.mp4',
    PROJECT_NAME='MyProject',
    # Параметры simple_background_extractor
    COLOR_SENSITIVITY_FACTOR=1,
    SIZE_SCALING_FACTOR=2,
    NUMBER_OF_FRAMES=250,

    # ObjectsCNNRecognizer
    CNNType='YOLO',
    NAME='coco1',
    DETECTION_MIN_CONFIDENCE=0.33,
    NUM_CLASSES=81,
    BATCH_SIZE=1,
    IMAGE_META_SIZE=93,

    BORDER_CROP=10,
)

ObjectTracker(**parameters)