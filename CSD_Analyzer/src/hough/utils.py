import cv2
import numpy.typing as npt
from skimage import io, color
from skimage.morphology import skeletonize
import os, shutil

output_folder = "./outputs/utils"

def thin_binary_image(filepath: str) -> npt.NDArray:
    """
    binary_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    thinning = cv2.ximgproc.thinningGuoHall(binary_image)
    cv2.imwrite(output_folder + "/thinning.png", thinning)
    
    return thinning
    """
    #if os.path.exists(output_folder):
    #    shutil.rmtree(output_folder)
    #os.mkdir(output_folder)


    binary_image = io.imread(filepath, as_gray=True)
    thinned_image = skeletonize(binary_image)
    thinned_image = (thinned_image * 255).astype('uint8')
    cv2.imwrite(output_folder + "/original.png", binary_image)
    cv2.imwrite(output_folder + "/thinning.png", thinned_image)

    return thinned_image

if __name__ == "__main__":
    """
    filepath = "./data/_archive/takahashi/thinning.png"
    thin_binary_image(filepath=filepath)
    """
    filepath_line = "./outputs/infer/class1.png"
    filepath_triangle = "./outputs/infer/class2.png"
    out = integrate_edges(filepath_line=filepath_line, filepath_triangle=filepath_triangle)
    thin_binary_image(out)
