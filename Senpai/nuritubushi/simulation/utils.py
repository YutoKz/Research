from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SimRange:
    """シミュレーションの範囲を表すクラス."""

    start: float
    end: float
    step: float

    def __len__(self) -> int:
        """シミュレーション範囲の離散分割されたサンプル数を返す."""
        sim_range = np.arange(self.start, self.end, self.step)

        return len(sim_range)

    def get_array(self) -> npt.NDArray[np.float64]:
        """シミュレーション範囲をnumpy配列で返す."""
        return np.arange(self.start, self.end, self.step)
