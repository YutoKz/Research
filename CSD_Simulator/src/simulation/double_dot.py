from functools import lru_cache
from typing import Literal

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter  # type: ignore  # noqa: PGH003
import random

from .utils import SimRange
from .utils import to_grayscale, add_noise

BorderType = Literal["3", "4", "5"]

Coord = tuple[float, float]


class ClassicDoubleQuantumDot:
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
        v_s: float = 0.0,  # v_dは0として考えている
    ) -> None:
        self.c_0 = c_0
        self.c_1 = c_1
        self.c_01 = c_01
        self.c_gate0_0 = c_gate0_0
        self.c_gate0_1 = c_gate0_1
        self.c_gate1_0 = c_gate1_0
        self.c_gate1_1 = c_gate1_1
        self.e = e
        self.v_s = v_s

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
        thickness: float = 0.1,
        salt_prob: float = 0.0,
        pepper_prob: float = 0.0,
        random_prob: float = 0.0, 
        gaussian: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """CSDをシミュレーションする関数. 塗りつぶしなし

        Args:
            range_v0 (npt.NDArray[np.float64]): v0の範囲
            range_v1 (npt.NDArray[np.float64]): v1の範囲
            
            thickness: 直線の太さ
            salt_prob: salt ノイズの割合
            pepper_prob: pepper ノイズの割合
            random_prob: random ノイズの割合
            gaussian: ガウシアンフィルタの標準偏差

        Returns:
            npt.NDArray[np.float64]: CSDのヒートマップ
        """
        original_csd = np.zeros((len(range_v1), len(range_v0)))
        #thickness = 0.1

        for i, v1 in enumerate(range_v0):
            for j, v0 in enumerate(range_v1):
                original_csd[i, j] = self._calculate_CSD_heatmap(v0, v1, thickness)

        # ノイズを追加
        output_csd: npt.NDArray[np.float64] = add_noise(original_csd, salt_prob=salt_prob, pepper_prob=pepper_prob, random_prob=random_prob)  

        # ガウシアンフィルタ
        output_csd: npt.NDArray[np.float64] = gaussian_filter(output_csd, sigma=gaussian)  # type: ignore  # noqa: PGH003

        return original_csd, output_csd

    def simulation_CSD_fill(
        self,
        range_v0: SimRange,
        range_v1: SimRange,
        width: int = 2,    # これは変更しない方がいいかも
        intensity_background: float = 0.0,
        intensity_line: float = 1.0,
        intensity_triangle: float = 1.0,
        salt_prob: float = 0.0,
        pepper_prob: float = 0.0,
        random_prob: float = 0.0, 
        gaussian: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """CSDをシミュレーションする関数. 塗りつぶしあり

        Args:
            range_v0 (npt.NDArray[np.float64]): v0の範囲
            range_v1 (npt.NDArray[np.float64]): v1の範囲
            
            width: 直線の太さ

            intensity_background: 背景の強度 (0.0 ~ 1.0)
            intensity_line: 直線の強度 (0.0 ~ 1.0)
            intensity_triangle: 三角形の強度 (0.0 ~ 1.0)

            salt_prob: salt ノイズの割合
            pepper_prob: pepper ノイズの割合
            random_prob: random ノイズの割合
            gaussian: ガウシアンフィルタの標準偏差
        
        Returns:
            npt.NDArray[np.float64]: CSDのヒートマップ
        """
        csd_img = Image.fromarray(np.zeros((len(range_v1), len(range_v0))))  # type: ignore  # noqa: PGH003
        draw = ImageDraw.Draw(csd_img)

        for n_1 in range(7):
            for n_0 in range(7):
                for border_type in ("3", "4", "5"):
                    match self._get_line(range_v0, range_v1, n_0, n_1, border_type):
                        case None:
                            continue
                        case start, end:
                            draw.line([start, end], fill=1, width=width)

                    triangle_1, triangel_2 = self._get_triangle(range_v0, range_v1, n_0, n_1)

                    draw.polygon(triangle_1, fill=2)
                    draw.polygon(triangel_2, fill=2)

        # ラベル (背景：0  直線：1  三角形：2) 
        label_csd = np.array(csd_img)

        # 強度情報をもとにグレースケール化
        grayscale_csd: npt.NDArray[np.float64] = to_grayscale(
            label_csd, 
            intensity_background=intensity_background, 
            intensity_line=intensity_line,
            intensity_triangle=intensity_triangle,
        )
        
        # ノイズ付与
        noisy_csd: npt.NDArray[np.float64] = add_noise(
            grayscale_csd, 
            salt_prob=salt_prob, 
            pepper_prob=pepper_prob, 
            random_prob=random_prob,
        )  

        # ガウシアンフィルタ
        output_csd: npt.NDArray[np.float64] = gaussian_filter(noisy_csd, sigma=gaussian)  # type: ignore  # noqa: PGH003

        return label_csd, output_csd

    def _get_line(
        self,
        sim_range_v0: SimRange,
        sim_range_v1: SimRange,
        n_0: int,
        n_1: int,
        border_num: BorderType,
    ) -> tuple[Coord, Coord] | None:
        v0_start, v0_end = self.__range_v0(n_0, n_1, border_num)

        start_index_v0 = self._scale(v0_start, sim_range_v0)
        end_index_v0 = self._scale(v0_end, sim_range_v0)

        v1_start = self.__func_border(v0_start, n_0, n_1, border_num)
        v1_end = self.__func_border(v0_end, n_0, n_1, border_num)

        start_index_v1 = self._scale(v1_start, sim_range_v1)
        end_index_v1 = self._scale(v1_end, sim_range_v1)

        return ((start_index_v0, start_index_v1), (end_index_v0, end_index_v1))

    def _get_triangle(
        self,
        sim_range_v0: SimRange,
        sim_range_v1: SimRange,
        n_0: int,
        n_1: int,
    ) -> tuple[tuple[Coord, Coord, Coord], tuple[Coord, Coord, Coord]]:
        _, start_3_v0 = self.__range_v0(n_0, n_1, "3")
        start_3_v1 = self.__func_border(start_3_v0, n_0, n_1, "3")

        start_4_v0, end_4_v0 = self.__range_v0(n_0, n_1, "4")
        start_4_v1 = self.__func_border(start_4_v0, n_0, n_1, "4")

        _, end_5_v0 = self.__range_v0(n_0 - 1, n_1, "5")
        end_5_v1 = self.__func_border(end_5_v0, n_0 - 1, n_1, "5")

        triangle_1 = (
            (self._scale(start_3_v0, sim_range_v0), self._scale(start_3_v1, sim_range_v1)),
            (self._scale(start_4_v0, sim_range_v0), self._scale(start_4_v1, sim_range_v1)),
            (self._scale(end_5_v0, sim_range_v0), self._scale(end_5_v1, sim_range_v1)),
        )

        end_4_v1 = self.__func_border(end_4_v0, n_0, n_1, "4")

        start_5_v0, _ = self.__range_v0(n_0, n_1, "5")
        start_5_v1 = self.__func_border(start_5_v0, n_0, n_1, "5")

        start_3_v0, _ = self.__range_v0(n_0 + 1, n_1 - 1, "3")
        start_3_v1 = self.__func_border(start_3_v0, n_0 + 1, n_1 - 1, "3")

        triangle_2 = (
            (self._scale(end_4_v0, sim_range_v0), self._scale(end_4_v1, sim_range_v1)),
            (self._scale(start_5_v0, sim_range_v0), self._scale(start_5_v1, sim_range_v1)),
            (self._scale(start_3_v0, sim_range_v0), self._scale(start_3_v1, sim_range_v1)),
        )

        return triangle_1, triangle_2

    def _scale(self, value: float, sim_range: SimRange) -> float:
        return (value - sim_range.start) / sim_range.step

    def _calculate_CSD_heatmap(self, v_0: float, v_1: float, thickness: float) -> float:
        for n_1 in range(5):  # TODO: 並列化したい
            for n_0 in range(5):
                n_0_start, n_0_end = self._range_3_v0(n_0, n_1)
                if n_0_start <= v_0 <= n_0_end and self.__distance_border_3(v_0, v_1, n_0, n_1) <= thickness:
                    return 1

                n_0_start, n_0_end = self._range_4_v0(n_0, n_1)
                if n_0_start <= v_0 <= n_0_end and self.__distance_border_4(v_0, v_1, n_0, n_1) <= thickness:
                    return 1

                n_0_start, n_0_end = self._range_5_v0(n_0, n_1)
                if n_0_start <= v_0 <= n_0_end and self.__distance_border_5(v_0, v_1, n_0, n_1) <= thickness:
                    return 1

        return 0

    def __func_border(self, v_0: float, n_0: int, n_1: float, border_num: BorderType) -> float:
        match border_num:
            case "3":
                return self._func_border_3(v_0, n_0, n_1)
            case "4":
                return self._func_border_4(v_0, n_0, n_1)
            case "5":
                return self._func_border_5(v_0, n_0, n_1)

    def __func_border_3(self, v_0: float, n_0: int, n_1: int) -> float:
        return (
            -self.__alpha_02 / self.__alpha_03 * v_0
            + self.e
            * (1 / self.__c_tilde_01 * n_1 + 1 / self.__c_omega_0 * (n_0 - 1 / 2) + self.v_s / self.e)
            / self.__alpha_03
        )

    def __distance_border_3(self, v_0: float, v_1: float, n_0: int, n_1: int) -> float:
        # TODO: このあたりをキャッシュ化orして高速にしたい
        a = self.__alpha_02 / self.__alpha_03
        b = 1
        c = (
            -self.e
            * (1 / self.__c_tilde_01 * n_1 + 1 / self.__c_omega_0 * (n_0 - 1 / 2) + self.v_s / self.e)
            / self.__alpha_03
        )

        top = np.abs(a * v_0 + b * v_1 + c)
        bottom = np.sqrt(a**2 + b**2)

        return top / bottom

    def __func_border_4(self, v_0: float, n_0: int, n_1: int) -> float:
        return (
            -self.__alpha_12 / self.__alpha_13 * v_0
            + self.e * (1 / self.__c_tilde_01 * n_0 + 1 / self.__c_omega_1 * (n_1 - 1 / 2)) / self.__alpha_13
        )

    def __distance_border_4(self, v_0: float, v_1: float, n_0: int, n_1: int) -> float:
        a = self.__alpha_12 / self.__alpha_13
        b = 1
        c = -self.e * (1 / self.__c_tilde_01 * n_0 + 1 / self.__c_omega_1 * (n_1 - 1 / 2)) / self.__alpha_13

        top = np.abs(a * v_0 + b * v_1 + c)
        bottom = np.sqrt(a**2 + b**2)

        return top / bottom

    def __func_border_5(self, v_0: float, n_0: int, n_1: int) -> float:
        return (self.__alpha_12 - self.__alpha_02) / (self.__alpha_03 - self.__alpha_13) * v_0 + self.e / (
            self.__alpha_03 - self.__alpha_13
        ) * (
            n_1 / self.__c_tilde_01
            - (n_0 + 1) / self.__c_tilde_01
            + (n_0 + 1 / 2) / self.__c_omega_0
            - (n_1 - 1 / 2) / self.__c_omega_1
        )

    def __distance_border_5(self, v_0: float, v_1: float, n_0: int, n_1: int) -> float:
        a = (self.__alpha_12 - self.__alpha_02) / (self.__alpha_03 - self.__alpha_13)
        b = -1
        c = (
            self.e
            / (self.__alpha_03 - self.__alpha_13)
            * (
                n_1 / self.__c_tilde_01
                - (n_0 + 1) / self.__c_tilde_01
                + (n_0 + 1 / 2) / self.__c_omega_0
                - (n_1 - 1 / 2) / self.__c_omega_1
            )
        )

        top = np.abs(a * v_0 + b * v_1 + c)
        bottom = np.sqrt(a**2 + b**2)

        return top / bottom

    def __range_v0(self, n_0: int, n_1: int, border_num: BorderType) -> tuple[float, float]:
        match border_num:
            case "3":
                return self._range_3_v0(n_0, n_1)
            case "4":
                return self._range_4_v0(n_0, n_1)
            case "5":
                return self._range_5_v0(n_0, n_1)

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
                + 2.0 * self.__alpha_13 * self.__c_omega_0 * self.__c_omega_1 * self.__c_tilde_01 * self.v_s / self.e
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
                - self.__alpha_03 * self.__c_omega_0 * self.__c_omega_1 * self.__c_tilde_01 * self.v_s / self.e
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
            intersection_4_top - self.__alpha_13 * self.__c_omega_1 * self.__c_tilde_01 * self.e * 0.5
        ) / intersection_4_bottom

        intersection_4_1_v0 = (
            intersection_4_top
            + self.e
            * 0.5
            * (
                -2 * self.__alpha_13 * self.__c_omega_0 * self.__c_omega_1
                + self.__alpha_13 * self.__c_omega_1 * self.__c_tilde_01
            )
            + self.__alpha_13 * self.__c_omega_0 * self.__c_omega_1 * self.__c_tilde_01 * self.v_s
        ) / intersection_4_bottom

        return intersection_4_3_v0, intersection_4_1_v0

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
            intersection_5_top - 2 * self.__alpha_13 * self.__c_omega_0 * self.__c_omega_1 * self.e * 0.5
        ) / intersection_5_bottom
        intersection_5_1_v0 = (
            intersection_5_top
            - self.__alpha_03 * self.__c_omega_0 * self.__c_omega_1 * self.e
            - self.__alpha_03 * self.__c_omega_0 * self.__c_omega_1 * self.__c_tilde_01 * self.v_s
            + self.__alpha_13 * self.__c_omega_0 * self.__c_omega_1 * self.__c_tilde_01 * self.v_s
        ) / intersection_5_bottom

        return intersection_4_5_v0, intersection_5_1_v0
