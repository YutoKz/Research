"""
    出力のフォルダ名をnoisy -> originalに変更してから動作未確認
"""

# type: ignore
import os
import shutil
import numpy as np
import pandas as pd
from simulation.double_dot import ClassicDoubleQuantumDot
from simulation.utils import SimRange
import random

import cv2

output_folder = "./output_data"

def create_dataset() -> None:
    """ 学習データセットを用意する関数.
    
    """
    # make directories
    if os.path.exists(output_folder + "/original"):
        shutil.rmtree(output_folder + "/original")

    if os.path.exists(output_folder + "/label"):
        shutil.rmtree(output_folder + "/label")
    
    os.mkdir(output_folder + "/original")
    os.mkdir(output_folder + "/label")

    # Attention: need to be fixed!
    # range of v0 / v1
    #range_v0 = np.arange(0, 10, 0.1) #元0, 10, 0.05
    #range_v1 = np.arange(0, 10, 0.1)
    range_v0 = SimRange(0, 10, 0.1)
    range_v1 = SimRange(0, 10, 0.1)

    # number of training data
    num_pattern = 5

    random.seed(0)

    parameter_list = []

    for i in range(num_pattern):
        if i % 100 == 0:
            print(i)
        for j in range(3):
            # DQD parameter
            if j == 0:
                c_01 = random.uniform(-0.1, -0.35)
            elif j == 1:
                c_01 = random.uniform(-0.1, -0.35)
            else:
                c_01 = random.uniform(-0.1, -0.35)
            c_gate0_0 = c_gate1_1 = random.uniform(-0.3, -0.25)      # 拡大縮小
            c_gate0_1 = c_gate1_0 = -0.08    # 傾き
            c_0 = -(c_01 + c_gate0_0 + c_gate1_0)
            c_1 = -(c_01 + c_gate0_1 + c_gate1_1)
            e = 1.0                                 # 拡大縮小          2.0~
            v_s = 0.7                               # 線形 / 非線形

            # CSD parameter 
            #thickness = 0.1
            width = 2
            intensity_background = 0.45
            intensity_line = 0.55
            intensity_triangle = 0.65
            salt_prob = 0.0
            pepper_prob = 0.0
            random_prob = 0.0
            gaussian = 2.0

            parameter_list.append([
                                3*i+j,
                                c_0, c_1, c_01, c_gate0_0, c_gate0_1, c_gate1_0,c_gate1_1, e, v_s,
                                width, 
                                intensity_background, intensity_line, intensity_triangle, 
                                salt_prob, pepper_prob, random_prob, 
                                gaussian,   
                            ]
            )

            # DQD
            dqd = ClassicDoubleQuantumDot(
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
            label_csd, noisy_csd = dqd.simulation_CSD_fill(
                range_v0=range_v0, 
                range_v1=range_v1, 
                width=width, 
                intensity_background=intensity_background,
                intensity_line=intensity_line,
                intensity_triangle=intensity_triangle,
                salt_prob=salt_prob,
                pepper_prob=pepper_prob,
                random_prob=random_prob,
                gaussian=gaussian,
            )

            

            # 注意！　逆三角形実装まで、一時的にnp.flip() ⇒ np.rot90()に変更中
            # noiseless CSD
            label_csd_gray = label_csd * 100
            cv2.imwrite(output_folder + f"/label/{3*i+j}.png", np.rot90(label_csd)) #np.flip(label_csd, axis=0)
            cv2.imwrite(output_folder + f"/label/{3*i+j}_gray.png", np.rot90(label_csd_gray)) #np.flip(label_csd_gray, axis=0)
            # noisy CSD
            noisy_csd_gray = noisy_csd * 255
            #cv2.imwrite(output_folder + f"/noisy/{3*i+j}.png", np.rot90(noisy_csd)) #np.flip(noisy_csd, axis=0)
            cv2.imwrite(output_folder + f"/original/{3*i+j}.png", np.rot90(noisy_csd_gray)) #np.flip(noisy_csd_gray, axis=0)
            
    df = pd.DataFrame(
        parameter_list, 
        columns=[
            "index", 
            "c_0", "c_1", "c_01", "c_gate0_0", "c_gate0_1", "c_gate1_0","c_gate1_1", "e", "v_s",
            "width", 
            "intensity_background", "intensity_line", "intensity_triangle", 
            "salt_prob", "pepper_prob", "random_prob", 
            "gaussian", 
        ]
    )
    df.sort_values(by="index").to_csv(output_folder + "/parameters.csv", index=False)
            
            
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