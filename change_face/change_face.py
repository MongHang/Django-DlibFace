import numpy as np
# import matplotlib.pyplot as plt
import cv2
import dlib
import math
import os


def rotate(image, angle, center=None, scale=1.0):
    # 取畫面寬高
    (h, w) = image.shape[:2]

    # 若中心點為無時，則中心點取影像的中心點
    if center is None:
        center = (w / 2, h / 2)

    # 產生旋轉矩陣Ｍ(第一個參數為旋轉中心，第二個參數旋轉角度，第三個參數：縮放比例)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 透過旋轉矩陣進行影像旋轉
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def face(img_jpg):
    # img = cv2.imread(r"./media/Uploaded Files/{}".format(img_jpg))  # 不支持中文
    img = cv2.imdecode(np.fromfile(file=rf"./media/Uploaded Files/{img_jpg}", dtype=np.uint8), cv2.IMREAD_COLOR)  # 支持中文
    img_bg = cv2.imread(r"./change_face/1.jpg")
    img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2BGRA)

    # 讀取人臉辨識模型
    detector = dlib.get_frontal_face_detector()
    # 讀取人臉辨識之特徵模型
    predictor = dlib.shape_predictor(r"./change_face/shape_predictor_68_face_landmarks.dat")

    location_dlib = detector(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 取得人臉輪廓
    for face in location_dlib:
        landmarks = predictor(img_gray, face)
        face_cut = []
        for i in range(28):
            if i <= 17:
                face_cut.append([landmarks.part(i).x, landmarks.part(i).y])
            elif 18 <= i < 27:
                face_cut.insert(17, [landmarks.part(i).x, landmarks.part(i).y - 10])

    # 取得xy邊界
    face_cut_x = []
    face_cut_y = []
    for i in range(len(face_cut)):
        face_cut_x.append(face_cut[i][0])
        face_cut_y.append(face_cut[i][1])

    face_cut_x_min = np.min(face_cut_x)
    face_cut_y_min = np.min(face_cut_y)
    face_cut_x_max = np.max(face_cut_x)
    face_cut_y_max = np.max(face_cut_y)

    face_cut_x_center = int((face_cut_x_min + face_cut_x_max) / 2)
    face_cut_y_center = int((face_cut_y_min + face_cut_y_max) / 2)

    face_center = np.array([face_cut_x_center, face_cut_y_center])

    face_cut = np.array([face_cut])

    face_add = (face_cut * 0.9 + face_center * 0.1).astype(int)  # 邊緣模糊(10%)

    mask = np.zeros(img.shape[:2], np.uint8)
    mask2 = np.zeros(img.shape[:2], np.uint8)

    cv2.polylines(mask, face_cut, 1, 255)
    cv2.fillPoly(mask, face_cut, 255)
    cv2.polylines(mask2, face_add, 1, 255)
    cv2.fillPoly(mask2, face_add, 255)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    img2 = img.copy()
    img4 = img.copy()
    blur_ = int((img.shape[0] + img.shape[1]) / 2 / 70)  # 模糊化 = 像素/70
    img2 = cv2.blur(img2, (blur_, blur_))

    mask2 = 255 - mask2
    mask6 = cv2.bitwise_and(mask, mask2)
    mask2 = 255 - mask2
    img2 = cv2.bitwise_and(img2, img2, mask=mask6)
    img4 = cv2.bitwise_and(img4, img4, mask=mask2)
    img3 = cv2.add(img2, img4)

    bg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    img3 = img3 + bg  # 背景黑轉白

    # 旋轉角度 = 鼻子角度
    right_nose = [landmarks.part(31).x, landmarks.part(31).y]
    left_nose = [landmarks.part(35).x, landmarks.part(35).y]

    nose_angle = float(-math.atan((right_nose[1] - left_nose[1]) / (right_nose[0] - left_nose[0])) * 180 / 3.14159)

    img3 = rotate(img3, -nose_angle)

    # 旋轉後的臉部邊界(切割用)
    x0 = landmarks.part(0).x - img3.shape[1] / 2
    y0 = landmarks.part(0).y - img3.shape[0] / 2
    x19 = landmarks.part(19).x - img3.shape[1] / 2
    y19 = landmarks.part(19).y - img3.shape[0] / 2
    x16 = landmarks.part(16).x - img3.shape[1] / 2
    y16 = landmarks.part(16).y - img3.shape[0] / 2
    x8 = landmarks.part(8).x - img3.shape[1] / 2
    y8 = landmarks.part(8).y - img3.shape[0] / 2

    nose_angle = nose_angle * 3.1415926 / 180  # 轉rad
    face_cut_x_min = int(math.cos(nose_angle) * x0 - math.sin(nose_angle) * y0 + img3.shape[1] / 2)
    face_cut_y_min = int(math.sin(nose_angle) * x19 + math.cos(nose_angle) * y19 + img3.shape[0] / 2)
    face_cut_x_max = int(math.cos(nose_angle) * x16 - math.sin(nose_angle) * y16 + img3.shape[1] / 2)
    face_cut_y_max = int(math.sin(nose_angle) * x8 + math.cos(nose_angle) * y8 + img3.shape[0] / 2)

    # 原圖切割
    img3 = img3[face_cut_y_min:face_cut_y_max, face_cut_x_min:face_cut_x_max]

    # 臉與熊貓不同向=水平翻轉
    bg_left_right = 1  # 1=熊貓向右
    face_left_right = (landmarks.part(27).x + landmarks.part(57).x) / 2 - landmarks.part(33).x

    if (face_left_right * bg_left_right) >= 0:
        img3 = np.flip(img3, axis=1)

    if img3.shape[1] >= img3.shape[0]:
        img3 = cv2.resize(img3, (200, int(200 * img3.shape[0] / img3.shape[1])))
    else:
        img3 = cv2.resize(img3, (int(200 * img3.shape[1] / img3.shape[0]), 200))

    # 熊貓人貼臉的起始座標(左上角)
    panda_x = int(220 - img3.shape[1] / 2)
    panda_y = int(200 - img3.shape[0] / 2)

    # 貼臉
    img_bg[panda_y:panda_y + img3.shape[0], panda_x:panda_x + img3.shape[1]] = img3

    # 轉灰階
    re_face = cv2.cvtColor(img_bg, cv2.COLOR_BGRA2GRAY)
    # print("img_jpg", img_jpg)

    # 輸出統一jpg
    new_name = f"re_{str(img_jpg).split('.')[0]}.jpg"
    # print("CF_new_name:", new_name)

    # 存檔
    # cv2.imwrite(rf"./media/New Face/{new_name}", re_face)  # 不支持中文
    cv2.imencode(".jpg", re_face)[1].tofile(rf"./media/New Face/{new_name}")  # 支持中文

    # print(os.getcwd())  # 查看工作目錄
