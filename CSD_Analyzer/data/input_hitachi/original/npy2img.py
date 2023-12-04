"""
    .npy形式のファイルから画像を出力
    (3, 縦, 横) で、(x座標, y座標, その座標での値) を指す形式のものを想定
"""

import cv2
import numpy as np
import numpy.typing as npt

filename = "./data/input_hitachi/2023-03-28_09_40_50.699913.npy"
#filename = "./data/input_hitachi/2023-03-23_15_10_40.163763.npy"

data = np.load(filename)

voltage_res_0 = data[0, 0, 1] - data[0, 0, 0]
voltage_res_1 = data[1, 1, 0] - data[1, 0, 0]

print(f"data shape: {data.shape}\n")
print("横軸: data[0] ")
print(f"|- number:   {data.shape[2]}")
print(f"|- range:    {data[0, 0, 0]} ~ {data[0, 0, data.shape[2]-1]}")
print(f"|- volt_res: {voltage_res_0}\n")
print("縦軸: data[1] ")
print(f"|- number:   {data.shape[1]}")
print(f"|- range:    {data[1, 0, 0]} ~ {data[1, data.shape[1]-1, 0]}")
print(f"|- volt_res: {voltage_res_1}\n")

array = data[2]

normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    
scaled_array = normalized_array * 255.0
    
uint8_array = scaled_array.astype(np.uint8)

print(f"output shape: {uint8_array.shape}")

cv2.imwrite("./data/input_hitachi/csd.png", np.flip(uint8_array, axis=0))