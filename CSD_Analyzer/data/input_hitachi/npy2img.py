import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

filename = "./data/input_hitachi/2023-03-28_09_40_50.699913.npy"


"""
data = np.load(filename)

# カラーマップを作成
plt.figure(figsize=(8, 8))
plt.imshow(data[2], cmap='hot', interpolation='nearest')
plt.axis('off')  # 軸の表示をオフにする

# 画像をNumPy配列に変換
plt.gcf().canvas.draw()
img_array = np.asarray(plt.gcf().canvas.renderer._renderer)

print(img_array.shape)

# Matplotlibのプロットを閉じる
plt.close()

# OpenCVで画像を保存
cv2.imwrite('csd.jpg.png', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
"""

#import pyarbtools
import matplotlib.pyplot as plt
import time
import numpy as np
import datetime 
import glob
import pandas as pd

B = np.load(filename)
print(B.shape)
print(B)

plt.figure(figsize=(24,18))
plt.rcParams["font.size"] = 16
plt.pcolormesh(B[0],B[1],B[2],shading='nearest',cmap='hot')
# plt.pcolormesh(np.array(result_ave)/Gain,shading='nearest',cmap='hot')
plt.title(filename,fontsize=15)
plt.colorbar()
plt.axis("equal")   # これなかったら意味ないのでは？？
#plt.clim([0, 10**-9.5])
plt.xlabel("SGS(V)",fontsize=20)
plt.ylabel("SG3(V)",fontsize=20)
plt.savefig("./data/input_hitachi/csd.png")