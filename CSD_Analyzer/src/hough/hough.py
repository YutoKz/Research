import cv2
import numpy as np
import pandas as pd
import os

from utils import integrate_edges

# あとからdetect_line_segmentのみ改良していったので、こっちを使う場合は注意
def detect_line(
    filepath: str,
    edge_extraction: bool = True,
    hough_threshold: int = 20,
) -> None:
    # フォルダ準備
    if os.path.exists("./data/output_hough") == False:
        os.mkdir("./data/output_hough")

    # 画像の読み込み
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("./data/output_hough/original.png", image)

    # ハフ変換による直線検出
    if edge_extraction:
        edges = cv2.Canny(image, 50, 100)
        cv2.imwrite("./data/output_hough/canny.png", edges)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=hough_threshold)
    else:
        lines = cv2.HoughLines(image, 1, np.pi / 180, threshold=hough_threshold)
    
    # 各直線に対する処理
    print(f"num of lines: {len(lines)}")
    lines_list = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        # 直線の切片と傾きを計算
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # ゼロ除算を防ぐために小さな値を追加
        intercept = y1 - slope * x1
        lines_list.append([slope, intercept])

        # 直線を描画
        line_color = (0, 0, 255) if slope > 1 else (0, 255, 0)
        cv2.line(rgb_image, (x1, y1), (x2, y2), line_color, 1)

    # 結果の保存
    cv2.imwrite("./data/output_hough/detected_lines.png", rgb_image)
    df = pd.DataFrame(lines_list, columns=["slope", "intercept"])
    df.to_csv("./data/output_hough/line_parapeters.csv", index=False)

def detect_line_segment(
    filepath: str,
    voltage_per_pixel: float = 1.0,
    edge_extraction: bool = True,
    hough_threshold: int = 10,
    hough_minLineLength: int = 10,
    hough_maxLineGap: int = 2,
) -> None:
    # フォルダ準備
    if os.path.exists("./data/output_hough") == False:
        os.mkdir("./data/output_hough")

    # 画像の読み込み
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("./data/output_hough/original.png", image)
    height, width = image.shape[:2]
    print(f"image size: ({height}, {width})")

    # ハフ変換による直線検出
    if edge_extraction:
        edges = cv2.Canny(image, 50, 100)
        cv2.imwrite("./data/output_hough/canny.png", edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold, minLineLength=hough_minLineLength, maxLineGap=hough_maxLineGap)
    else:
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=hough_threshold, minLineLength=hough_minLineLength, maxLineGap=hough_maxLineGap)
    
    # 各直線に対する処理
    print(f"Total: {len(lines)}")
    num_of_horizontal_lines = 0
    num_of_vertical_lines = 0
    num_of_interdot_lines = 0
    lines_list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 直線の切片と傾き
        # cv2では左上が原点, 横軸x縦軸yであることに注意
        slope_minus = (y2 - y1) / (x2 - x1 + 1e-6)  # ゼロ除算を防ぐ
        slope = slope_minus * -1
        intercept = (height - y1) - slope * x1

        # 直線を描画
        if slope < -1:
            # 赤
            line_color = (0, 0, 255)
            num_of_vertical_lines += 1
            lines_list.append(["vertical", slope, intercept * voltage_per_pixel])
        elif -1 <= slope <= 0:
            # 緑
            line_color = (0, 255, 0)
            num_of_horizontal_lines += 1
            lines_list.append(["horizontal",slope, intercept * voltage_per_pixel])
        else:
            # 青
            line_color = (255, 0, 0)
            num_of_interdot_lines += 1
            lines_list.append(["interdot", slope, intercept * voltage_per_pixel])
        cv2.line(rgb_image, (x1, y1), (x2, y2), line_color, 1)
    print(f"|- Horizontal: {num_of_horizontal_lines}\n|- Vertical:   {num_of_vertical_lines}\n|- Interdot:   {num_of_interdot_lines}")

    # 結果の保存
    cv2.imwrite("./data/output_hough/detected_lines.png", rgb_image)
    df = pd.DataFrame(lines_list, columns=["type", "slope", "intercept"])
    df.sort_values(by="type").to_csv("./data/output_hough/line_parapeters.csv", index=False)

if __name__ == "__main__":
    #filepath_line = "./data/output_infer/class1.png"
    #filepath_triangle = "./data/output_infer/class2.png"
    #filepath = integrate_edges(filepath_line, filepath_triangle)

    filepath = "./data/output_utils/small.png"
    
    detect_line_segment(
        filepath=filepath, 
        edge_extraction=False,
        hough_threshold=30,         
        hough_minLineLength=10,     
        hough_maxLineGap=2              
    )
    
