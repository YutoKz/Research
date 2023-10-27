# type: ignore
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from simulation.double_dot import DoubleQuantumDot
import random

import cv2

def create_dataset() -> None:
    """ 学習データセットを用意する関数.
    
    """
    # make directories
    if os.path.exists("./data/noisy"):
        shutil.rmtree("./data/noisy")

    if os.path.exists("./data/original"):
        shutil.rmtree("./data/original")
    
    os.mkdir("./data/noisy")
    os.mkdir("./data/original")

    # Attention: need to be fixed!
    # range of v0 / v1
    range_v0 = np.arange(0, 10, 0.02) #元0, 10, 0.05
    range_v1 = np.arange(0, 10, 0.02)

    # number of training data
    num_pattern = 1000

    random.seed(0)

    for i in range(num_pattern):
        if i % 100 == 0:
            print(i)
        for j in range(3):
            # DQD parameter
            #c_01 = -0.1                                            # 構造 
            if j == 0:
                c_01 = random.uniform(-0.6, -0.4)
            elif j == 1:
                c_01 = random.uniform(-0.4, -0.01)
            else:
                c_01 = random.uniform(-0.01, -0.001)
            c_gate0_0 = c_gate1_1 = random.uniform(-0.3, -0.1)      # 拡大縮小
            c_gate0_1 = c_gate1_0 = random.uniform(-0.1, -0.001)    # 傾き
            c_0 = -(c_01 + c_gate0_0 + c_gate1_0)
            c_1 = -(c_01 + c_gate0_1 + c_gate1_1)
            e = 1.0                                 # 拡大縮小          2.0~
            v_s = 0.0                               # 線形 / 非線形

            # CSD parameter 
            thickness = 0.1
            salt_prob = 0.0
            pepper_prob = random.uniform(0.0, 0.2)
            random_prob = random.uniform(0.0, 0.5)
            gaussian = random.uniform(0.8, 1.2)


            # DQD
            dqd = DoubleQuantumDot(
                c_0=c_0,
                c_1=c_1,
                c_01=c_01,
                c_gate0_0=c_gate0_0,
                c_gate0_1=c_gate0_1,
                c_gate1_0=c_gate1_0,
                c_gate1_1=c_gate1_1,
                e=e,
                v_s=v_s,
            )

            # CSD
            original_csd, noisy_csd = dqd.simulation_CSD(
                range_v0=range_v0, 
                range_v1=range_v1, 
                thickness=thickness, 
                salt_prob=salt_prob,
                pepper_prob=pepper_prob,
                random_prob=random_prob,
                gaussian=gaussian, 
            )

            # original CSD
            original_csd_confirm = original_csd * 255
            cv2.imwrite(f"./data/original/{3*i+j}.png", np.flip(original_csd, axis=0))
            cv2.imwrite(f"./data/original/{3*i+j}_gray.png", np.flip(original_csd_confirm, axis=0))

            # noisy CSD
            noisy_csd_confirm = noisy_csd * 255
            cv2.imwrite(f"./data/noisy/{3*i+j}.png", np.flip(noisy_csd, axis=0))
            cv2.imwrite(f"./data/noisy/{3*i+j}_gray.png", np.flip(noisy_csd_confirm, axis=0))

if __name__ == "__main__":
    create_dataset()



"""
# original csd
plt.figure(figsize=(8, 8))
plt.imshow(
    original_csd,
    extent=[range_v0.min(), range_v0.max(), range_v1.min(), range_v1.max()],
    origin="lower",
    cmap="gray_r",
    aspect="auto",
)
plt.axis("off") 
plt.savefig(f"./data/original/{i}.png", bbox_inches="tight", pad_inches=0)
# noisy csd
plt.figure(figsize=(8, 8))
plt.imshow(
    noisy_csd,
    extent=[range_v0.min(), range_v0.max(), range_v1.min(), range_v1.max()],
    origin="lower",
    cmap="gray_r",
    aspect="auto",
)
plt.axis("off") 
plt.savefig(f"./data/noisy/{i}.png", bbox_inches="tight", pad_inches=0)
"""