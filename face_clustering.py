import os
import cv2
import math
import pandas as pd

import dlib


DIR_BASE_PATH = "C:/Users/SHPark/Desktop/Project/2021_datavoucher/beauty_hankook/" \
                "data/tarball-master/AFAD-Full.tar/AFAD-Full/"

TARGET_AGE = [str(i) for i in range(20, 30)]
TARGET_SEX = 112    # 111: 남성, 112: 여성
IMG_RESIZE_WIDTH = 512
IMG_RESIZE_HEIGHT = 512
LIMIT_N = -1

# landmark 범위 설정
N_FACE_POINTS = 68
FACE_POINTS_IDX = {'jaw_line': [i for i in range(0, 17)],
                   'left_eyebrow': [i for i in range(17, 22)],
                   'right_eyebrow': [i for i in range(22, 27)],
                   'nose_bridge': [i for i in range(27, 31)],
                   'lower_nose': [i for i in range(31, 36)],
                   'left_eye': [i for i in range(36, 42)],
                   'right_eye': [i for i in range(42, 48)],
                   'outer_lip': [i for i in range(48, 60)],
                   'inner_lip': [i for i in range(60, 68)]}

# 도출할 Feature 목록과 해당 Feature landmark
FEATURE_LANDMARK = {'faceWidth': [1, 15],
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


def get_crop_img(img, start_point, end_point):
    """
    :param img: (numpy.ndarray) 크롭할 이미지
    :param start_point: (list) 크롭할 부분의 좌측상단 좌표
    :param end_point: (list) 크롭할 부분의 우측하단 좌표
    :return: (numpy.ndarray) 크롭된 이미지
    """
    x_dist = end_point[0] - start_point[0]
    y_dist = end_point[1] - start_point[1]
    x_margin, y_margin = x_dist//4, y_dist//4   # 25% 마진

    c_sx, c_sy = start_point[0] - x_margin, start_point[1] - y_margin
    c_ex, c_ey = end_point[0] + x_margin, end_point[1] + y_margin

    c_sx, c_sy = max(c_sx, 0), max(c_sy, 0)
    c_ex, c_ey = min(c_ex, img.shape[1]), min(c_ey, img.shape[0])

    return img[c_sy: c_ey, c_sx: c_ex]


images, img_names = [], []
for age in TARGET_AGE:
    # 파일들 경로
    filesPath = DIR_BASE_PATH + age + '/' + str(TARGET_SEX)
    files = os.listdir(filesPath)
    print(age, len(files))

    # 이미지들 로드
    for i, file in enumerate(files):
        # 이미지 로드 제한
        if LIMIT_N != -1 and i > LIMIT_N:
            break
        # jpg 이미지만 로드
        if os.path.splitext(file)[1] == '.jpg':
            img_names.append(os.path.splitext(file)[0])
            image = cv2.imread(filesPath + '/' + file)
            images.append(image)

print('load image count: ', len(images))    # 로드한 이미지 수

# 라이브러리 객체화
detector = dlib.get_frontal_face_detector()     # face recognition
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")   # face landmark

featureDf = pd.DataFrame(columns=(['fileName'] + list(FEATURE_LANDMARK.keys())))     # Feature 저장할 df
for image, img_name in zip(images, img_names):
    # 이미지 원본
    OriginalImg = image

    # 이미지에서 얼굴들 추출
    tempImg = cv2.cvtColor(src=OriginalImg, code=cv2.COLOR_BGR2GRAY)
    faces = detector(tempImg)
    for face in faces:
        # 얼굴마다 크롭하여 크롭된 얼굴 사진 사이즈 통일
        cropOriginalImg = get_crop_img(OriginalImg,
                                       (face.left(), face.top()), (face.right(), face.bottom()))
        cropOriginalImg = cv2.resize(src=cropOriginalImg, dsize=(IMG_RESIZE_WIDTH, IMG_RESIZE_HEIGHT))
        sampleImg = cropOriginalImg.copy()

        # 크롭 이미지 내 얼굴 재서치
        imgGrey = cv2.cvtColor(src=cropOriginalImg, code=cv2.COLOR_BGR2GRAY)
        r_face = detector(imgGrey)
        # 얼굴 탐색 실패시
        if not r_face:
            continue
        r_face = r_face[0]

        # 얼굴 박스 좌표 추출
        x1, y1 = r_face.left(), r_face.top()
        x2, y2 = r_face.right(), r_face.bottom()

        # 얼굴 박스 그리기
        cv2.rectangle(sampleImg, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        # 얼굴 포인트 탐색
        landmarks = predictor(image=imgGrey, box=r_face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # 얼굴 포인트 그리기
            cv2.circle(img=sampleImg, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)

        # Feature 도출
        face_feature = {}
        for k, v in FEATURE_LANDMARK.items():
            x1, y1 = landmarks.part(v[0]).x, landmarks.part(v[0]).y
            x2, y2 = landmarks.part(v[1]).x, landmarks.part(v[1]).y
            face_feature[k] = math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))

            cv2.line(img=sampleImg, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=1, lineType=cv2.LINE_8)

        # Feature 추가
        face_feature['fileName'] = img_name
        featureDf = featureDf.append(face_feature, ignore_index=True)

print(featureDf.shape[0])
featureDf.to_csv('./result/face_features.csv')
