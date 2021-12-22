import os
import cv2
import random

from face_classification import FaceTypeClassifier


filesPath = './test'
files = os.listdir(filesPath)
print('load image count: ', len(files))  # 로드한 이미지 수
img_ls = []
for file in files:
    # jpg 이미지만 로드
    if os.path.splitext(file)[1] == '.jpg':
        image = cv2.imread(filesPath + '/' + file)
        print(type(image))
        img_ls.append(image)

ftc = FaceTypeClassifier()
faces, face_type = ftc.classifier(img_ls)
for f, f_t in zip(faces, face_type):
    cv2.imshow("Face Image", f)
    cv2.waitKey()
    print(f_t)
    # file_name = './result/test_result/' + str(f_t) + '/' + str(int(random.uniform(1e7, 9.9999999e7))) + '.jpg'
    # cv2.imwrite(file_name, f)
