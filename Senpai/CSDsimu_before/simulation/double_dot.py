from functools import lru_cache

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter  # type: ignore  # noqa: PGH003


class DoubleQuantumDot:
    """DQD(double quantum dot)を表現するクラス.

    量子ドットと周辺との間のキャパシタンスをコンストラクタに与える。
    教科書(Semiconductor Nanostructures)の19章を元に設計
    """

    def __init__(
        self,
        c_0: float,
        c_1: float,
        c_01: float,
        c_gate0_0: float,
        c_gate0_1: float,
        c_gate1_0: float,
        c_gate1_1: float,
        e: float,
    ) -> None:
        self.c_0 = c_0
        self.c_1 = c_1
        self.c_01 = c_01
        self.c_gate0_0 = c_gate0_0
        self.c_gate0_1 = c_gate0_1
        self.c_gate1_0 = c_gate1_0
        self.c_gate1_1 = c_gate1_1
        self.e = e

        c_omega_temp = 1 - c_01**2 / (c_0 * c_1)
        self.__c_omega_0 = c_omega_temp * c_0
        self.__c_omega_1 = c_omega_temp * c_1

        c_det = c_0 * c_1 - c_01**2

        self.__c_tilde_01 = -c_det / c_01

        self.__alpha_02 = (c_01 * c_gate0_1 - c_1 * c_gate0_0) / c_det
        self.__alpha_12 = (c_01 * c_gate0_0 - c_0 * c_gate0_1) / c_det
        self.__alpha_03 = (c_01 * c_gate1_1 - c_1 * c_gate1_0) / c_det
        self.__alpha_13 = (c_01 * c_gate1_0 - c_0 * c_gate1_1) / c_det

        # キャッシュのための処理
        self._func_border_3 = lru_cache(maxsize=100)(self.__func_border_3)
        self._func_border_4 = lru_cache(maxsize=100)(self.__func_border_4)
        self._func_border_5 = lru_cache(maxsize=100)(self.__func_border_5)
        self._range_3_v0 = lru_cache(maxsize=100)(self.__range_3_v0)
        self._range_4_v0 = lru_cache(maxsize=100)(self.__range4_v0)
        self._range_5_v0 = lru_cache(maxsize=100)(self.__range5_v0)

    def simulation_CSD(
        self,
        range_v0: npt.NDArray[np.float64],
        range_v1: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """CSDをシミュレーションする関数.

        Args:
            range_v0 (npt.NDArray[np.float64]): v0の範囲
            range_v1 (npt.NDArray[np.float64]): v1の範囲

        Returns:
            npt.NDArray[np.float64]: CSDのヒートマップ
        """
        csd = np.zeros((len(range_v1), len(range_v0)))
        thickness = 0.1

        for i, v1 in enumerate(range_v0):
            for j, v0 in enumerate(range_v1):
                csd[i, j] = self._calculate_CSD_heatmap(v0, v1, thickness)

        blur_csd: npt.NDArray[np.float64] = gaussian_filter(csd, sigma=1.0)  # type: ignore  # noqa: PGH003

        return blur_csd

    def _calculate_CSD_heatmap(self, v_0: float, v_1: float, thickness: float) -> float:
        for n_1 in range(6):
            for n_0 in range(6):
                n_0_start, n_0_end = self._range_3_v0(n_0, n_1)
                if n_0_start <= v_0 <= n_0_end and np.abs(v_1 - self._func_border_3(v_0, n_0, n_1)) <= thickness:
                    return 1

                n_0_start, n_0_end = self._range_4_v0(n_0, n_1)
                if n_0_start <= v_0 <= n_0_end and np.abs(v_1 - self._func_border_4(v_0, n_0, n_1)) <= thickness:
                    return 1

                n_0_start, n_0_end = self._range_5_v0(n_0, n_1)
                if n_0_start <= v_0 <= n_0_end and np.abs(v_1 - self._func_border_5(v_0, n_0, n_1)) <= thickness:
                    return 1

        return 0

    def __func_border_3(self, v_0: float, n_0: int, n_1: int) -> float:
        return (
            -self.__alpha_02 / self.__alpha_03 * v_0
            + self.e * (1 / self.__c_tilde_01 * n_1 + 1 / self.__c_omega_0 * (n_0 - 1 / 2)) / self.__alpha_03
        )

    def __func_border_4(self, v_0: float, n_0: int, n_1: int) -> float:
        return (
            -self.__alpha_12 / self.__alpha_13 * v_0
            + self.e * (1 / self.__c_tilde_01 * n_0 + 1 / self.__c_omega_1 * (n_1 - 1 / 2)) / self.__alpha_13
        )

    def __func_border_5(self, v_0: float, n_0: int, n_1: int) -> float:
        return (self.__alpha_12 - self.__alpha_02) / (self.__alpha_03 - self.__alpha_13) * v_0 + self.e / (
            self.__alpha_03 - self.__alpha_13
        ) * (
            n_1 / self.__c_tilde_01
            - (n_0 + 1) / self.__c_tilde_01
            + (n_0 + 1 / 2) / self.__c_omega_0
            - (n_1 - 1 / 2) / self.__c_omega_1
        )

    def __range_3_v0(self, n_0: int, n_1: int) -> tuple[float, float]:
        """(19.4)式 (教科書参照) のv_0の範囲を計算する関数.

        Args:
            n_0 (int): _description_
            n_1 (int): _description_

        Returns:
            tuple[float, float]: _description_
        """
        intersection_3_bottom = (
            self.__c_omega_0
            * self.__c_omega_1
            * self.__c_tilde_01
            * (self.__alpha_02 * self.__alpha_13 - self.__alpha_03 * self.__alpha_12)
        )
        intersection_3_top = (
            self.e
            * 0.5
            * (
                -2 * n_0 * self.__alpha_03 * self.__c_omega_0 * self.__c_omega_1
                + 2 * n_0 * self.__alpha_13 * self.__c_omega_1 * self.__c_tilde_01
                - 2 * n_1 * self.__alpha_03 * self.__c_omega_0 * self.__c_tilde_01
                + 2.0 * n_1 * self.__alpha_13 * self.__c_omega_0 * self.__c_omega_1
                + self.__alpha_03 * self.__c_omega_0 * self.__c_tilde_01
                - self.__alpha_13 * self.__c_omega_1 * self.__c_tilde_01
            )
        )

        intersection_3_4_v0 = intersection_3_top / intersection_3_bottom
        intersection_3_6_v0 = (
            intersection_3_top
            + self.e
            * (
                -self.__alpha_03 * self.__c_omega_0 * self.__c_tilde_01
                + self.__alpha_03 * self.__c_omega_0 * self.__c_omega_1
            )
        ) / intersection_3_bottom

        return intersection_3_6_v0, intersection_3_4_v0

    def __range4_v0(self, n_0: int, n_1: int) -> tuple[float, float]:
        intersection_4_bottom = (
            self.__c_omega_0
            * self.__c_omega_1
            * self.__c_tilde_01
            * (self.__alpha_02 * self.__alpha_13 - self.__alpha_03 * self.__alpha_12)
        )
        intersection_4_top = (
            self.e
            * 0.5
            * (
                -2 * n_0 * self.__alpha_03 * self.__c_omega_0 * self.__c_omega_1
                + 2 * n_0 * self.__alpha_13 * self.__c_omega_1 * self.__c_tilde_01
                - 2 * n_1 * self.__alpha_03 * self.__c_omega_0 * self.__c_tilde_01
                + 2.0 * n_1 * self.__alpha_13 * self.__c_omega_0 * self.__c_omega_1
                + self.__alpha_03 * self.__c_omega_0 * self.__c_tilde_01
            )
        )

        intersection_4_3_v0 = (
            intersection_4_top - self.__alpha_13 * self.__c_omega_1 * self.__c_tilde_01
        ) / intersection_4_bottom
        intersection_4_5_v0 = (
            intersection_4_top
            - 2 * self.__alpha_13 * self.__c_omega_0 * self.__c_omega_1
            + self.__alpha_13 * self.__c_omega_1 * self.__c_tilde_01
        ) / intersection_4_bottom

        return intersection_4_3_v0, intersection_4_5_v0

    def __range5_v0(self, n_0: int, n_1: int) -> tuple[float, float]:
        intersection_5_bottom = (
            self.__c_omega_0
            * self.__c_omega_1
            * self.__c_tilde_01
            * (self.__alpha_02 * self.__alpha_13 - self.__alpha_03 * self.__alpha_12)
        )
        intersection_5_top = (
            self.e
            * 0.5
            * (
                -2 * n_0 * self.__alpha_03 * self.__c_omega_0 * self.__c_omega_1
                + 2 * n_0 * self.__alpha_13 * self.__c_omega_1 * self.__c_tilde_01
                - 2 * n_1 * self.__alpha_03 * self.__c_omega_0 * self.__c_tilde_01
                + 2.0 * n_1 * self.__alpha_13 * self.__c_omega_0 * self.__c_omega_1
                + self.__alpha_03 * self.__c_omega_0 * self.__c_tilde_01
                + self.__alpha_13 * self.__c_omega_1 * self.__c_tilde_01
            )
        )

        intersection_4_5_v0 = (
            intersection_5_top - 2 * self.__alpha_13 * self.__c_omega_0 * self.__c_omega_1
        ) / intersection_5_bottom
        intersection_5_1_v0 = (
            intersection_5_top - 2 * self.__alpha_03 * self.__c_omega_0 * self.__c_omega_1
        ) / intersection_5_bottom

        return intersection_4_5_v0, intersection_5_1_v0
