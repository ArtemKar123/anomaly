import cv2
import numpy as np
from imutils import face_utils
from retinaface.pre_trained_models import get_model as get_retina_model
import dlib
import torch
from utils.utils import IoUfrom2bboxes, crop_face


class Landmarks:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retina_model = get_retina_model("resnet50_2020-07-20", max_size=512, device=self.device)
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor('./weights/shape_predictor_81_face_landmarks.dat')

    def retina_landmarks(self, image):
        '''

        :param image: BGR image
        :param model: retina model
        :return:
        '''

        height, width, _ = image.shape

        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.retina_model.predict_jsons(frame)
        if len(faces) == 0:
            return None

        landmarks = []
        size_list = []
        for face_idx in range(len(faces)):
            x0, y0, x1, y1 = faces[face_idx]['bbox']
            landmark = np.array([[x0, y0], [x1, y1]] + faces[face_idx]['landmarks'])
            face_s = (x1 - x0) * (y1 - y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]]

        return landmarks

    def dlib_landmarks(self, image):
        '''

        :param image: BGR image
        :param face_detector: dlib face detector
        :param face_predictor: dlib face predictor
        :return:
        '''

        height, width, _ = image.shape

        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = self.face_detector(frame, 1)

        if len(faces) == 0:
            return None

        landmarks = []
        size_list = []
        for face_idx in range(len(faces)):
            landmark = self.face_predictor(frame, faces[face_idx])
            landmark = face_utils.shape_to_np(landmark)
            x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
            x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
            face_s = (x1 - x0) * (y1 - y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]]

        return landmarks


def reorder_landmark(landmark):
    landmark_add = np.zeros((13, 2))
    for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
        landmark_add[idx] = landmark[idx_l]
    landmark[68:] = landmark_add
    return landmark


if __name__ == '__main__':
    image = cv2.imread('sample2.jpeg')
    cv2.imshow('a', image)
    lm = Landmarks()
    bboxes = lm.retina_landmarks(image)
    landmark = lm.dlib_landmarks(image)[0]
    bbox_lm = np.array(
        [landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()])

    iou_max = -1
    for i in range(len(bboxes)):
        iou = IoUfrom2bboxes(bbox_lm, bboxes[i].flatten())
        if iou_max < iou:
            bbox = bboxes[i]
            iou_max = iou
    landmark = reorder_landmark(landmark)
    img, landmark, bbox, __ = crop_face(image, landmark, bbox, margin=True, crop_by_bbox=False)

    cv2.imshow('b', img)
    cv2.waitKey()
