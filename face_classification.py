import cv2
import math
import pandas as pd
import pickle

import dlib


class FaceTypeClassifier:
    def __init__(self):
        # 이미지 크롭 사이즈
        self.IMG_RESIZE_WIDTH = 512
        self.IMG_RESIZE_HEIGHT = 512

        # 얼굴 추출 후 마진 (100/Margin * 100%) = 25%
        self.MARGIN = 4

        # 모델 path
        self.MODEL_PATH = './model/'
        self.MODEL_NAME = 'model.pickle'

        # 얼굴 Feature 위치
        self.FEATURE_LANDMARK = {'faceWidth': [1, 15],
                                 'noseWidth': [31, 35],
                                 'noseHeight': [27, 30],
                                 'lipInnerWidth': [60, 64],
                                 'lipOuterWidth': [48, 54],
                                 'lipOuterHeight': [51, 57],
                                 'leftEyeWidth': [36, 39],
                                 'rightEyeWidth': [42, 45],
                                 'chinWidth': [6, 10],
                                 'leftEyebrowToEye': [19, 37],
                                 'rightEyebrowToEye': [24, 44],
                                 'betweenTheEyebrows': [39, 42],
                                 'leftJawLine': [9, 13],
                                 'rightJawLine': [3, 7],
                                 'philanthropyHeight': [33, 51]
                                 }

        # 라이브러리 객체화
        self.detector = dlib.get_frontal_face_detector()     # face recognition
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")   # face landmark

    # 모델을 로드하는 메서드
    def _load_model(self):
        """
        :return: 학습된 얼굴 분류 모델
        """
        f = open(self.MODEL_PATH + self.MODEL_NAME, 'rb')
        model = pickle.load(f)
        f.close()
        return model

    # 이미지 크롭 메서드
    def _get_crop_img(self, img, start_point, end_point):
        """
        :param img: (numpy.ndarray) 크롭할 이미지
        :param start_point: (list) 크롭할 부분의 좌측상단 좌표
        :param end_point: (list) 크롭할 부분의 우측하단 좌표
        :return: (numpy.ndarray) 크롭된 이미지
        """
        x_dist = end_point[0] - start_point[0]
        y_dist = end_point[1] - start_point[1]
        x_margin, y_margin = x_dist//self.MARGIN, y_dist//self.MARGIN   # 25% 마진

        c_sx, c_sy = start_point[0] - x_margin, start_point[1] - y_margin
        c_ex, c_ey = end_point[0] + x_margin, end_point[1] + y_margin

        c_sx, c_sy = max(c_sx, 0), max(c_sy, 0)
        c_ex, c_ey = min(c_ex, img.shape[1]), min(c_ey, img.shape[0])

        return img[c_sy: c_ey, c_sx: c_ex]

    # 얼굴형 분류 메서드
    def classifier(self, images):
        """
        :param images: (list[numpy.ndarray]) 얼굴형을 분류할 이미지들
        :return: (list[numpy.ndarray]), (list[int]) 분류된 이미지,  분류된 얼굴 타입
        """
        # model load
        k_means = self._load_model()

        res_face_type, res_face = [], []
        for img in images:
            # 이미지 원본
            original_img = img

            # 이미지에서 얼굴들 추출
            temp_img = cv2.cvtColor(src=original_img, code=cv2.COLOR_BGR2GRAY)
            faces = self.detector(temp_img)
            for face in faces:
                # 얼굴마다 크롭하여 크롭된 얼굴 사진 사이즈 통일
                crop_original_img = self._get_crop_img(original_img,
                                                       (face.left(), face.top()), (face.right(), face.bottom()))
                crop_original_img = cv2.resize(src=crop_original_img,
                                               dsize=(self.IMG_RESIZE_WIDTH, self.IMG_RESIZE_HEIGHT))
                sample_img = crop_original_img.copy()
                # cv2.imshow("Face Original Image", crop_original_img)
                # cv2.waitKey()

                # 크롭 이미지 내 얼굴 재서치
                img_grey = cv2.cvtColor(src=crop_original_img, code=cv2.COLOR_BGR2GRAY)
                rect_face = self.detector(img_grey)

                # 얼굴 탐색 실패시
                if not rect_face:
                    continue
                rect_face = rect_face[0]

                # 얼굴 박스 좌표 추출
                x1, y1 = rect_face.left(), rect_face.top()
                x2, y2 = rect_face.right(), rect_face.bottom()

                # 얼굴 박스 그리기
                cv2.rectangle(sample_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

                # 얼굴 포인트 탐색
                landmarks = self.predictor(image=img_grey, box=rect_face)
                for n in range(68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y

                    # 얼굴 포인트 그리기
                    cv2.circle(img=sample_img, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)

                # cv2.imshow("Face Image", sample_img)
                # cv2.waitKey()

                # Face Feature 도출
                face_feature = {}
                for k, v in self.FEATURE_LANDMARK.items():
                    x1, y1 = landmarks.part(v[0]).x, landmarks.part(v[0]).y
                    x2, y2 = landmarks.part(v[1]).x, landmarks.part(v[1]).y
                    face_feature[k] = math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))

                    # 주요 포인트 연결 (Face Feature 추출)
                    cv2.line(img=sample_img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0),
                             thickness=1, lineType=cv2.LINE_8)

                # Face Feature 값으로 Face type 도출
                feature_df = pd.DataFrame(columns=self.FEATURE_LANDMARK.keys())
                feature_df = feature_df.append(face_feature, ignore_index=True)
                res_face_type.append(k_means.predict(feature_df)[0])
                res_face.append(crop_original_img)

        return res_face, res_face_type
