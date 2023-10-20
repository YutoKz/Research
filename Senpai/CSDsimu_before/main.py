import matplotlib.pyplot as plt
import numpy as np
from simulation.double_dot import DoubleQuantumDot


def main() -> None:
    c_01 = -0.1
    c_gate0_0 = -0.8
    c_gate0_1 = -0.1
    c_gate1_0 = -0.1
    c_gate1_1 = -0.8
    c_0 = -(c_01 + c_gate0_0 + c_gate1_0)
    c_1 = -(c_01 + c_gate0_1 + c_gate1_1)
    dqd = DoubleQuantumDot(
        c_0=c_0,
        c_1=c_1,
        c_01=c_01,
        c_gate0_0=c_gate0_0,
        c_gate0_1=c_gate0_1,
        c_gate1_0=c_gate1_0,
        c_gate1_1=c_gate1_1,
        e=2,
    )

    range_v0 = np.arange(0, 10, 0.05)
    range_v1 = np.arange(0, 10, 0.05)
    csd = dqd.simulation_CSD(range_v0=range_v0, range_v1=range_v1)

    plt.figure(figsize=(8, 8))
    plt.imshow(
        csd,
        extent=[range_v0.min(), range_v0.max(), range_v1.min(), range_v1.max()],
        origin="lower",
        cmap="gray_r",
        aspect="auto",
    )
    plt.title("CSD")
    plt.xlabel("v0")
    plt.ylabel("v1")
    plt.grid(True)
    plt.savefig("csd.png")


if __name__ == "__main__":
    main()
