# type: ignore
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal
from simulation.double_dot import ClassicDoubleQuantumDot, SimRange
from simulation.double_dot_hubbard import HubbardDoubleQuantumDot
from simulation.utils import SimRange

MethodType = Literal["classic", "hubbard"]


def simulate_CSD(method: MethodType):
    range_v0 = SimRange(0, 10, 0.05)
    range_v1 = SimRange(0, 10, 0.05)

    match method:
        case "classic":
            c_01 = -0.1
            c_gate0_0 = -0.8
            c_gate0_1 = -0.1
            c_gate1_0 = -0.1
            c_gate1_1 = -0.8
            c_0 = -(c_01 + c_gate0_0 + c_gate1_0)
            c_1 = -(c_01 + c_gate0_1 + c_gate1_1)
            dqd = ClassicDoubleQuantumDot(
                c_0=c_0,
                c_1=c_1,
                c_01=c_01,
                c_gate0_0=c_gate0_0,
                c_gate0_1=c_gate0_1,
                c_gate1_0=c_gate1_0,
                c_gate1_1=c_gate1_1,
                e=1,
                v_s=0.07,
            )

            # csd = dqd.simulation_CSD(range_v0=range_v0, range_v1=range_v1)
            csd = dqd._draw_CSD_heatmap(range_v0=range_v0, range_v1=range_v1)

        case "hubbard":
            dqd = HubbardDoubleQuantumDot(
                u_0=6.1, u_1=6.1, u_01=2.5, t=0.3, j_e=0.0, j_p=0.0, j_t0=0.0, j_t1=0.0, e=1.60
            )

            csd = dqd.simulation_CSD(range_v0=range_v0, range_v1=range_v1, disable_progress_bar=False)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        csd,
        extent=[range_v0.start, range_v0.end, range_v1.start, range_v1.end],
        origin="lower",
        cmap="gray_r",
        aspect="auto",
    )
    ax.set_title("CSD", fontsize=20)
    ax.set_xlabel("v0", fontsize=20)
    ax.set_ylabel("v1", fontsize=20)
    ax.grid()
    ax.tick_params(labelsize=14)
    fig.savefig("csd.png")

    np.savetxt("csd.csv", csd, delimiter=",")


def main() -> None:
    simulate_CSD(method="hubbard")


if __name__ == "__main__":
    main()
