import numpy as np
import cv2

img = cv2.imread("3_452_120_label.png")
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
cv2.imwrite("output.png", output)
