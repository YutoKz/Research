# type: ignore
import matplotlib.pyplot as plt
import numpy as np
from simulation.double_dot import DoubleQuantumDot


def main() -> None:

    # DQD parameter
    c_01 = -0.1                             # 構造              ~ -3.0 ~ -0.05 ~ -0.00001 
    c_gate0_0 = -0.8                        # 拡大縮小          -0.8 ~
    c_gate0_1 = -0.1                        # 傾き              -0.2 ~ -0.001
    c_gate1_0 = -0.1                        # 傾き              -0.2 ~ -0.001
    c_gate1_1 = -0.8                        # 拡大縮小          -0.8 ~
    c_0 = -(c_01 + c_gate0_0 + c_gate1_0)
    c_1 = -(c_01 + c_gate0_1 + c_gate1_1)
    e = 2.0                                 # 拡大縮小          2.0~
    v_s = 0.0                               # 線形 / 非線形

    # CSD parameter 
    thickness = 0.05
    salt_prob = 0.0
    pepper_prob = 0.2
    random_prob = 0.45
    gaussian = 1.2

    # range of v0 / v1
    range_v0 = np.arange(0, 10, 0.1) #元0, 10, 0.05
    range_v1 = np.arange(0, 10, 0.1)


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

    # 描画
    plt.figure(figsize=(8, 8))
    plt.imshow(
        original_csd,
        extent=[range_v0.min(), range_v0.max(), range_v1.min(), range_v1.max()],
        origin="lower",
        cmap="gray_r",
        aspect="auto",
    )
    plt.axis("off") #個人的に追加
    #plt.title("CSD")
    #plt.xlabel("v0")
    #plt.ylabel("v1")
    #plt.grid(True)
    plt.savefig("./result/original_csd.png", bbox_inches="tight", pad_inches=0)

    plt.figure(figsize=(8, 8))
    plt.imshow(
        noisy_csd,
        extent=[range_v0.min(), range_v0.max(), range_v1.min(), range_v1.max()],
        origin="lower",
        cmap="gray_r",
        aspect="auto",
    )
    plt.axis("off") #個人的に追加
    #plt.title("CSD")
    #plt.xlabel("v0")
    #plt.ylabel("v1")
    #plt.grid(True)
    plt.savefig("./result/noisy_csd.png", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
