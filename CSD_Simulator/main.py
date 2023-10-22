# type: ignore
import matplotlib.pyplot as plt
import numpy as np
from simulation.double_dot import DoubleQuantumDot


def main() -> None:

    # parameter
    c_01 = -0.1
    c_gate0_0 = -0.8
    c_gate0_1 = -0.1
    c_gate1_0 = -0.1
    c_gate1_1 = -0.8
    c_0 = -(c_01 + c_gate0_0 + c_gate1_0)
    c_1 = -(c_01 + c_gate0_1 + c_gate1_1)

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
        e=7,
        v_s=0.0,
    )

    original_csd, noisy_csd = dqd.simulation_CSD(
        range_v0=range_v0, 
        range_v1=range_v1, 
        thickness=0.05, 
        pepper=0.2,
        gaussian=1.0, 
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
