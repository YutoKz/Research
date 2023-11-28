# type: ignore
import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import Literal
from simulation.double_dot import ClassicDoubleQuantumDot, SimRange
#from simulation.double_dot_hubbard import HubbardDoubleQuantumDot
from simulation.utils import SimRange


def main() -> None:

    # DQD parameter
    c_01 = -0.1                           # 構造              ~ -0.5 ~ -0.05 ~ -0.00001 
    c_gate0_0 = -0.2                        # 拡大縮小          -0.8 ~ -0.3
    c_gate0_1 = -0.01                       # 傾き              -0.1 ~ -0.001
    c_gate1_0 = -0.01                        # 傾き              -0.1 ~ -0.001
    c_gate1_1 = -0.2                        # 拡大縮小          -0.8 ~  -0.3
    c_0 = -(c_01 + c_gate0_0 + c_gate1_0)
    c_1 = -(c_01 + c_gate0_1 + c_gate1_1)
    e = 1.0                                 # 拡大縮小          2.0~
    v_s = 0.7                               # 線形 / 非線形

    # CSD parameter 
    width = 2
    intensity_background = 0.37,
    intensity_line = 0.45,
    intensity_triangle = 0.6,
    salt_prob = 0.0
    pepper_prob = 0.0
    random_prob = 0.0
    gaussian = 2.0

    # range
    range_v0 = SimRange(0, 10, 0.1)
    range_v1 = SimRange(0, 10, 0.1)

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
    """
    original_csd, noisy_csd = dqd.simulation_CSD(
        range_v0=range_v0.get_array(), # main.py内のrange_v0の定義をSimRangeに書き換えた関係で少し変更済み 
        range_v1=range_v1.get_array(), 
        thickness=thickness, 
        salt_prob=salt_prob,
        pepper_prob=pepper_prob,
        random_prob=random_prob,
        gaussian=gaussian, 
    )
    """
    original_csd, noisy_csd = dqd.simulation_CSD_fill(
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

    # original CSD
    original_csd_confirm = original_csd * 127
    cv2.imwrite(f"./result/label.png", np.flip(original_csd, axis=0))
    cv2.imwrite(f"./result/label_gray.png", np.flip(original_csd_confirm, axis=0))
    # noisy CSD
    noisy_csd_confirm = noisy_csd * 255
    cv2.imwrite(f"./result/noisy.png", np.flip(noisy_csd, axis=0))
    cv2.imwrite(f"./result/noisy_gray.png", np.flip(noisy_csd_confirm, axis=0))
    



    """
    # 描画
    v0_min = range_v0.get_array().min()
    v0_max = range_v0.get_array().max()
    v1_min = range_v1.get_array().min()
    v1_max = range_v1.get_array().max()

    plt.figure(figsize=(8, 8))
    plt.imshow(
        original_csd,
        extent=[v0_min, v0_max, v1_min, v1_max],
        origin="lower",
        cmap="gray_r",
        aspect="auto",
    )
    plt.axis("off") #個人的に追加
    #plt.grid(True)
    plt.savefig("./result/original_csd.png", bbox_inches="tight", pad_inches=0)

    plt.figure(figsize=(8, 8))
    plt.imshow(
        noisy_csd,
        extent=[v0_min, v0_max, v1_min, v1_max],
        origin="lower",
        cmap="gray_r",
        aspect="auto",
    )
    plt.axis("off") #個人的に追加
    #plt.grid(True)
    plt.savefig("./result/noisy_csd.png", bbox_inches="tight", pad_inches=0)
    """

if __name__ == "__main__":
    main()
