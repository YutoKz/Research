import cv2
import numpy.typing as npt
from skimage import io, color
from skimage.morphology import skeletonize
import os, shutil

output_folder = "./data/output_utils"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.mkdir(output_folder)

def integrate_edges(
    filepath_line: str,
    filepath_triangle: str,
) -> str:
    """ 三角形の輪郭と直線を合わせる

    Args:
        filepath_line:      CSDの直線の二値画像
        filepath_triangle:  CSDの三角形の二値画像
        
    """
    filepath_output = output_folder + "/integrated_edge.png"

    img_line = cv2.imread(filepath_line, cv2.IMREAD_GRAYSCALE)
    img_triangle = cv2.imread(filepath_triangle, cv2.IMREAD_GRAYSCALE)

    edges_triangle = cv2.Canny(img_triangle, 50, 100)

    output = cv2.add(img_line, edges_triangle)

    cv2.imwrite(filepath_output, output)

    return filepath_output

def thin_binary_image(filepath: str) -> npt.NDArray:
    """
    binary_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    thinning = cv2.ximgproc.thinningGuoHall(binary_image)
    cv2.imwrite(output_folder + "/thinning.png", thinning)
    
    return thinning
    """
    binary_image = io.imread(filepath, as_gray=True)
    thinned_image = skeletonize(binary_image)
    thinned_image = (thinned_image * 255).astype('uint8')
    cv2.imwrite(output_folder + "/original.png", binary_image)
    cv2.imwrite(output_folder + "/thinning.png", thinned_image)

    return thinned_image

if __name__ == "__main__":
    filepath = "./data/_archive/takahashi/192_192.png"

    thin_binary_image(filepath=filepath)