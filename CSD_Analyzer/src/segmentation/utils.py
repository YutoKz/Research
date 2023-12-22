import cv2
import numpy as np
import torch

# fix seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def integrate_edges(
    filepath_line: str,
    filepath_triangle: str,
    filepath_output: str,
) -> str:
    """ 三角形の輪郭と直線を合わせる

    Args:
        filepath_line:      CSDの直線の二値画像
        filepath_triangle:  CSDの三角形の二値画像
        
    """

    img_line = cv2.imread(filepath_line, cv2.IMREAD_GRAYSCALE)
    img_triangle = cv2.imread(filepath_triangle, cv2.IMREAD_GRAYSCALE)

    edges_triangle = cv2.Canny(img_triangle, 50, 100)

    output = cv2.add(img_line, edges_triangle)

    cv2.imwrite(filepath_output, output)

    return filepath_output

if __name__ == "__main__":
    integrate_edges("./outputs/infer/pred_class1.png", "./outputs/infer/pred_class2.png", filapath_output="./outputs/infer/integrated_edge.png")