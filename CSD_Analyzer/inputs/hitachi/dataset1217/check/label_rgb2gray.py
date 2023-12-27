"""
    手動でラベル付けした画像をRGBで読み込み、各画像に対して
        学習に使う形式                  (0, 1,   2,   ..)
        確認用に画素値を上げた形式       (0, 100, 200, ..)
    の二つを生成する。
"""

import numpy as np
import cv2

for i in range(7):
    img = cv2.imread("./check/" + str(i) + ".png")
    output = np.zeros_like(img)

    red = np.array([0, 0, 255])
    blue = np.array([255, 0, 0])
    label_1 = np.array([1, 1, 1])
    label_2 = np.array([2, 2, 2])

    red_indices = np.all(img == red, axis=-1)
    blue_indices = np.all(img == blue, axis=-1)

    output[red_indices] = label_1
    output[blue_indices] = label_2

    print(np.unique(output))
    cv2.imwrite("./label/" + str(i) + ".png", output)
    cv2.imwrite("./check/" + str(i) + "_gray.png", output*100)

