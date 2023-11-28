import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter  # type: ignore  # noqa: PGH003
import random
from dataclasses import dataclass

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

def to_grayscale(image, intensity_background, intensity_line, intensity_triangle):
    """入力画像(ラベル形式)を、各要素の強度情報をもとにグレースケール化する。ただし値は0.0~1.0とする。"""
    grayscale_csd = np.copy(image)
    grayscale_csd[grayscale_csd == 0] = intensity_background
    grayscale_csd[grayscale_csd == 1] = intensity_line
    grayscale_csd[grayscale_csd == 2] = intensity_triangle
    
    return grayscale_csd

def add_noise(image, salt_prob, pepper_prob, random_prob) -> npt.NDArray[np.float64]:
    """入力画像にノイズを付与する。"""
    noisy_image = np.copy(image)

    total_pixels = image.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)
    num_random = int(total_pixels * random_prob)

    # add Salt noise 
    salt_coordinates = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coordinates[0], salt_coordinates[1]] = 1.0

    # add Pepper noise
    pepper_coordinates = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coordinates[0], pepper_coordinates[1]] = 1.0

    # add Random noise
    random_coordinates = [np.random.randint(0, i - 1, num_random) for i in image.shape]
    noisy_image[random_coordinates[0], random_coordinates[1]] = [random.random() for i in range(num_random)]

    return noisy_image

