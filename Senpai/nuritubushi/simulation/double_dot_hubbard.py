import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .utils import SimRange


class HubbardDoubleQuantumDot:
    """Hubbard模型を用いてCSDのシミュレーションを行うクラス."""

    def __init__(
        self,
        u_0: float,
        u_1: float,
        u_01: float,
        t: float,
        j_e: float,
        j_p: float,
        j_t0: float,
        j_t1: float,
        e: float = 1.0,
    ) -> None:
        self.u_0 = u_0
        self.u_1 = u_1
        self.u_01 = u_01
        self.t = t
        self.j_e = j_e
        self.j_p = j_p
        self.j_t0 = j_t0
        self.j_t1 = j_t1
        self.e = e

        # 何回も再計算するのは無駄なので保存しておく
        self.__alpha_0 = (u_1 - u_01) * u_0 / (u_0 * u_1 - u_01**2)
        self.__alpha_1 = (u_0 - u_01) * u_1 / (u_0 * u_1 - u_01**2)
        self.__beta_0 = 1 - self.__alpha_0
        self.__beta_1 = 1 - self.__alpha_1

    def simulation_CSD(
        self,
        range_v0: SimRange,
        range_v1: SimRange,
        *,
        disable_progress_bar: bool = True,
    ) -> npt.NDArray[np.float64]:
        """CSDをシミュレーションする関数.

        Args:
            range_v0 (npt.NDArray[np.float64]): v0の範囲
            range_v1 (npt.NDArray[np.float64]): v1の範囲
            disable_progress_bar (bool, optional): プログレスバーを表示するかどうか. Defaults to False.

        Returns:
            npt.NDArray[np.float64]: CSDのヒートマップ
        """
        csd = np.zeros((len(range_v1), len(range_v0)))
        for i, v0 in tqdm(enumerate(range_v0.get_array()), disable=disable_progress_bar):
            for j, v1 in enumerate(range_v1.get_array()):
                value = self._calculate_ground_state(v0, v1)

                # (↑, ↓), (↓, ↑)を区別しないための暫定的な処置
                if value == 7:  # noqa: PLR2004
                    csd[i, j] = 6
                else:
                    csd[i, j] = value

        return csd

    def _calculate_ground_state(self, v0: float, v1: float) -> int:
        # constant energy shiftは無視
        mu_0 = self.e * (self.__alpha_0 * v0 + self.__beta_0 * v1)
        mu_1 = self.e * (self.__alpha_1 * v1 + self.__beta_1 * v0)

        h = self.calculate_hamiltonian(mu_0, mu_1)

        w, v = np.linalg.eig(h)

        ground_index = np.argmin(w)
        ground_state = v[ground_index]

        # 線型結合された状態のうち、最も確率が高い状態のindexを返す
        return int(np.argmax(ground_state**2))

    def calculate_hamiltonian(self, mu_0: float, mu_1: float) -> npt.NDArray[np.float64]:
        """Generalized Hubbard modelのハミルトニアンを定義する関数.

        それぞれのドットの電子数が0~2の場合のハミルトニアン

        Args:
            mu_0 (float): ドット0の電気化学ポテンシャル
            mu_1 (float): ドット1の電気化学ポテンシャル
        """
        # E_Bは仮
        E_B = 0.0

        H_1 = 0.0
        H_2 = np.array([[-mu_0 + E_B, -self.t], [-self.t, -mu_1 + E_B]])
        H_3 = np.array([[-mu_0 - E_B, -self.t], [-self.t, -mu_1 - E_B]])
        H_4 = -mu_0 - mu_1 + self.u_01 - self.j_e + 2 * E_B
        H_5 = np.array(
            [
                [-mu_0 - mu_1 + self.u_01, self.j_e, -self.t - self.j_t0, -self.t - self.j_t1],
                [self.j_e, -mu_0 - mu_1 + self.u_01, -self.t - self.j_t0, -self.t - self.j_t1],
                [-self.t - self.j_t0, -self.t - self.j_t0, -2 * mu_0 + self.u_0, self.j_p],
                [-self.t - self.j_t1, -self.t - self.j_t1, self.j_p, -2 * mu_1 + self.u_1],
            ],
        )
        H_6 = -mu_0 - mu_1 + self.u_01 - self.j_e - 2 * E_B
        H_7 = np.array(
            [
                [-2 * mu_0 - mu_1 + self.u_0 + 2 * self.u_01 - self.j_e + E_B, -self.t - self.j_t0 - self.j_t1],
                [-self.t - self.j_t0 - self.j_t1, -2 * mu_1 - mu_0 + self.u_1 + 2 * self.u_01 - self.j_e + E_B],
            ],
        )
        H_8 = np.array(
            [
                [-2 * mu_0 - mu_1 + self.u_0 + 2 * self.u_01 - self.j_e - E_B, -self.t - self.j_t0 - self.j_t1],
                [-self.t - self.j_t0 - self.j_t1, -2 * mu_1 - mu_0 + self.u_1 + 2 * self.u_01 - self.j_e - E_B],
            ],
        )
        H_9 = -2 * mu_0 - 2 * mu_1 + self.u_0 + self.u_1 + 4 * self.u_01 - 2 * self.j_e

        H: npt.NDArray[np.float64] = np.block(
            [  # type: ignore  # noqa: PGH003
                [H_1, np.zeros((1, 15))],
                [np.zeros((2, 1)), H_2, np.zeros((2, 13))],
                [np.zeros((2, 3)), H_3, np.zeros((2, 11))],
                [np.zeros((1, 5)), H_4, np.zeros((1, 10))],
                [np.zeros((4, 6)), H_5, np.zeros((4, 6))],
                [np.zeros((1, 10)), H_6, np.zeros((1, 5))],
                [np.zeros((2, 11)), H_7, np.zeros((2, 3))],
                [np.zeros((2, 13)), H_8, np.zeros((2, 1))],
                [np.zeros((1, 15)), H_9],
            ],
        )

        return H
