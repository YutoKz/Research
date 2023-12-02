import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import os, shutil

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
    if os.path.exists("./data/output_hough"):
        shutil.rmtree("./data/output_hough")
    os.mkdir("./data/output_hough")
    os.mkdir("./data/output_hough/individual_line")

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
    all_lines_image = np.copy(rgb_image)
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        
        # 直線の切片と傾き
        # cv2では左上が原点, 横軸x縦軸yであることに注意
        slope_minus = (y2 - y1) / (x2 - x1 + 1e-6)  # ゼロ除算を防ぐ
        slope = slope_minus * -1
        intercept = (height - y1) - slope * x1

        # 直線を描画
        one_line_image = np.copy(rgb_image)
        if slope < -1:
            # 赤
            line_color = (0, 0, 255)
            num_of_vertical_lines += 1
            lines_list.append(["vertical", slope, intercept * voltage_per_pixel])
            cv2.line(one_line_image, (x1, y1), (x2, y2), line_color, 1)
            cv2.imwrite(f"./data/output_hough/individual_line/vertical_{i}.png", one_line_image)
        elif -1 <= slope <= 0:
            # 緑
            line_color = (0, 255, 0)
            num_of_horizontal_lines += 1
            lines_list.append(["horizontal",slope, intercept * voltage_per_pixel])
            cv2.line(one_line_image, (x1, y1), (x2, y2), line_color, 1)
            cv2.imwrite(f"./data/output_hough/individual_line/horizontal_{i}.png", one_line_image)
        else:
            # 青
            line_color = (255, 0, 0)
            num_of_interdot_lines += 1
            lines_list.append(["interdot", slope, intercept * voltage_per_pixel])
            cv2.line(one_line_image, (x1, y1), (x2, y2), line_color, 1)
            cv2.imwrite(f"./data/output_hough/individual_line/interdot_{i}.png", one_line_image)
        cv2.line(all_lines_image, (x1, y1), (x2, y2), line_color, 1)
    print(f"|- Horizontal: {num_of_horizontal_lines}\n|- Vertical:   {num_of_vertical_lines}\n|- Interdot:   {num_of_interdot_lines}")

    # 結果の保存
    cv2.imwrite("./data/output_hough/detected_lines.png", all_lines_image)
    df = pd.DataFrame(lines_list, columns=["type", "slope", "intercept"])
    df.sort_values(by="type").to_csv("./data/output_hough/line_parapeters.csv", index=True)

# CSDの特徴を考慮し、傾きの範囲を3分割して個別にハフ変換するために実装
# 理想的には、異なる3つのtheta区間に一つずつ目的の傾きが存在するため、
# それぞれ異なる閾値を設定できるようにして確実に検出できるようにすることには、
# 投票数に大きな違いがあっても検出できるなど、一定の効果があるはずだが。。
def detect_peak_coordinate(
    hough_array: npt.NDArray,
    threshold: int,
    theta_gap: int,
) -> npt.NDArray:
    # 極大値を検出
    # cv2.HoughLinesを再現した以下のサイトを参考にした。
    # https://campkougaku.com/2021/08/17/hough3/#toc2
    hough_array[hough_array < threshold] = 0
 
    peak = (hough_array[1:-1, 1:-1] >= hough_array[0:-2, 1:-1]) * (hough_array[1:-1, 1:-1] >= hough_array[2:, 1:-1]) * \
           (hough_array[1:-1, 1:-1] >= hough_array[1:-1, 0:-2]) * (hough_array[1:-1, 1:-1] >= hough_array[1:-1, 2:]) * \
           (hough_array[1:-1, 1:-1] > 0)
    peak = np.array(np.where(peak)) + 1
    peak[1, :] += theta_gap
    
    return peak

def hough_transform(
    filepath: str, 
    rho_res: float, 
    theta_res: float = np.pi/180,
    threshold_vertical: int = 0,
    threshold_interdot: int = 0,
    threshold_horizontal: int = 0,
) -> None:
    """
    Args:
        filepath: 
        rho_res: ρの分解能
        theta_res: θの分解能 (rad)
    """
    # フォルダ準備
    if os.path.exists("./data/output_hough"):
        shutil.rmtree("./data/output_hough")
    os.mkdir("./data/output_hough")

    # 画像の読み込み
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("./data/output_hough/original.png", img)
    height, width = img.shape[:2]
    img = np.array(img)
    
    # エッジ座標
    y, x = np.where(img)

    # 各θごとのヒストグラムを保存
    hough = []
 
    #rhoの最大値は画像の対角
    rho_max = np.ceil(np.sqrt(img.shape[0] * img.shape[0] + img.shape[1] * img.shape[1]) / rho_res)
    rng = np.round(np.pi / theta_res)
 
    for i in np.arange(-rng, rng):
        rho2 = np.round((x * np.cos(i * theta_res) + y * np.sin(i * theta_res)) / rho_res)
        hist, _ = np.histogram(rho2, bins=np.arange(0, rho_max))
        hough.append(hist)
    
    # 縦軸：ρ, 横軸：θ　2次元配列
    hough_array = np.array(hough).T  # 転置して各列が異なる角度に対応
    
    # -pi ~ pi のρ-θ図を作成
    plt.imshow(hough_array, cmap='viridis', extent=[-rng, rng, 0, rho_max], aspect='auto', origin='lower')
    plt.colorbar(label='Frequency')
    plt.title('Hough Transform in rho - theta Space')
    plt.xlabel('Theta (radians)')
    plt.ylabel('Rho')
    plt.savefig("./data/output_hough/rho_theta.png")
    plt.close()
    
    
    # ρ-θ図を3つの範囲に分割
    tmp = np.copy(hough_array)
    tmp = np.split(tmp, 2, axis=1)
    hough_array_negative = tmp[0]                           # vertical
    hough_array_0_90 = np.split(tmp[1], 2, axis=1)[0]       # interdot
    hough_array_90_180 = np.split(tmp[1], 2, axis=1)[1]     # horizontal

    # 各領域で異なる閾値のもと投票数が極大値をとる座標を取得
    peak_negative = detect_peak_coordinate(hough_array_negative, threshold_vertical, 0)
    peak_0_90 = detect_peak_coordinate(hough_array_0_90, threshold_interdot, int(rng))
    peak_90_180 = detect_peak_coordinate(hough_array_90_180, threshold_horizontal, int(rng * 3 / 2))
    peak = np.hstack((peak_negative, peak_0_90, peak_90_180))

    # 得られた直線を元の画像に描画
    x1 = 0
    x2 = width - 1
    for i in range(peak.shape[1]):
        rho = peak[0, i] * rho_res
        theta = peak[1, i] * theta_res - np.pi
        if theta != 0 and theta != np.pi and theta != -np.pi:
            y1 = int(rho / np.sin(theta))
            y2 = int(y1 - x2 / np.tan(theta))
            if theta < 0: 
                line_color = (0, 0, 255)
            elif 0 <= theta <= np.pi/2:
                line_color = (255, 0, 0)
            else:
                line_color = (0, 255, 0) 
            cv2.line(rgb_img, (x1, y1), (x2, y2), line_color, 1)
        else:
            cv2.line(rgb_img, (int(rho), 0), (int(rho), height-1), (0, 0, 255), 1)
        
    cv2.imwrite("./data/output_hough/hough_transform.png", rgb_img)


if __name__ == "__main__":

    """
    filepath_line = "./data/output_train_simu/result/6_class1.png"
    filepath_triangle = "./data/output_train_simu/result/6_class2.png"
    filepath = integrate_edges(filepath_line, filepath_triangle)

   
    hough_transform(
        filepath=filepath,
        rho_res=0.5,
        theta_res=np.pi/180,
        threshold_vertical=25,
        threshold_interdot=25,
        threshold_horizontal=25,
    )
    """
    
    filepath = "./data/output_utils/small.png"
    
    
    detect_line_segment(
        filepath=filepath, 
        edge_extraction=False,
        hough_threshold=10,         
        hough_minLineLength=3,     
        hough_maxLineGap=3              
    )
    
    
