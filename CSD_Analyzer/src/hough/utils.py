import cv2
import numpy.typing as npt

output_folder = "./data/output_utils"

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
    binary_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    thinning = cv2.ximgproc.thinningGuoHall(binary_image)
    cv2.imwrite(output_folder + "/thinning.png", thinning)
    
    return thinning
