"""
    二値化した CSD から、Hough変換により直線のパラメータを抽出する。
"""

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import os, shutil
from typing import Literal

from utils import integrate_edges


output_folder = "./data/output_hough"
MethodType = Literal["slope_intercept", "slope"]


# あとからdetect_line_segmentのみ改良していったので、こっちを使う場合は注意
def detect_line(
    filepath: str,
    edge_extraction: bool = True,
    hough_threshold: int = 20,
) -> None:
    """ Hough変換( cv2.HoughLines() ) を用いた直線抽出。
    """
    # フォルダ準備
    if os.path.exists("./data/output_hough") == False:
        os.mkdir("./data/output_hough")

    # 画像の読み込み
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("./data/output_hough/original.png", image)

    # Hough変換による直線検出
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
    """ 一般化Hough変換( cv2.HoughLinesP() ) を用いた線分抽出。
    """
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

    # Hough変換による直線検出
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

def hough_transform(
    img,
    rho_res,
    theta_res
) -> npt.NDArray:
    """ Hough変換の結果を rho-theta 図で保存、2次元配列として返す。
    
    Args:
        img: 入力画像
        rho_res: rho 解像度 [px / px]
        theta_res: theta 解像度 [rad / px]

    Returns:
        npt.NDArray: Hough変換した結果。rho,thetaで表された直線の投票数を保存。(縦軸rho, 横軸theta)
    """
    # エッジ座標
    y, x = np.where(img)

    # 各θごとのヒストグラムを保存
    hough = []
 
    # rhoの最大値は画像の対角
    rho_max = np.ceil(np.sqrt(img.shape[0] * img.shape[0] + img.shape[1] * img.shape[1]) / rho_res)
    rng = np.round(np.pi / theta_res)
 
    for i in np.arange(-rng, rng):
        rho2 = np.round((x * np.cos(i * theta_res) + y * np.sin(i * theta_res)) / rho_res)
        hist, _ = np.histogram(rho2, bins=np.arange(0, rho_max))
        hough.append(hist)
    
    # 縦軸：ρ, 横軸：θ　2次元配列
    hough_array = np.array(hough).T  # 転置して各列が異なる角度に対応
    
    # -pi ~ pi のρ-θ図を作成
    plt.imshow(hough_array, cmap='viridis', extent=[-180, 180, 0, rho_max*rho_res], aspect='auto', origin='lower')
    plt.colorbar(label='Votes')
    plt.title('Hough Transform in rho - theta Space')
    plt.xlabel('Theta [degree]')
    plt.ylabel('Rho [pixel]')
    plt.savefig(output_folder + "/rho_theta.png")
    plt.close()

    return hough_array

def hough_transform_CSD(
    method: MethodType,
    filepath: str, 
    threshold_vertical: int = 0,
    threshold_interdot: int = 0,
    threshold_horizontal: int = 0,
    voltage_per_pixel: float = 1.0,
    rho_res: float = 0.5,
    theta_res: float = np.pi / 180,
) -> None:
    """ CSD の性質を活用したHough変換。異なるtheta領域を個別の閾値で直線検出

    Args:
        method: モード制御
        filepath: 入力画像

        threshold_vertical:                         縦          
        threshold_interdot:     直線検出用の閾値     ドット間     
        threshold_horizontal:                       横       

        voltage_per_pixel: 1pxあたりの電圧

        rho_res: rho 解像度 [px / px]
        theta_res: theta 解像度 [rad / px]
    """

    # フォルダ準備
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)

    # 画像の読み込み
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(output_folder + "/original.png", img)
    height, width = img.shape[:2]
    img = np.array(img)

    # ρ-θ図を3つの範囲に分割
    hough_array = hough_transform(img, rho_res, theta_res)    # 縦軸：ρ, 横軸：θ　2次元配列
    hough_array_vertical, hough_array_interdot, hough_array_horizontal = _split_hough_array(hough_array)
    
    theta_array = np.arange(hough_array.shape[1])
    theta_array_vertical, theta_array_interdot, theta_array_horizontal = _split_theta_array(theta_array)

    match method:
        case "slope_intercept":
            # 目標: 3つのtheta領域それぞれに対し個別に通常のHough変換による 直線(rho + theta) 抽出

            os.mkdir("./data/output_hough/individual_line")

            # 各領域で異なる閾値のもと、投票数が極大値をとる座標を取得
            peak_vertical = _detect_peak_coordinate(
                hough_array_vertical, 
                theta_array_vertical,
                threshold_vertical,
                #0
            )
            peak_interdot = _detect_peak_coordinate(
                hough_array_interdot, 
                theta_array_interdot,
                threshold_interdot, 
                #int(rng)
            )
            peak_horizontal = _detect_peak_coordinate(
                hough_array_horizontal,
                theta_array_horizontal, 
                threshold_horizontal, 
                #int(rng * 3 / 2)
            )
            peak = np.hstack((peak_vertical, peak_interdot, peak_horizontal))   # ここでつなげず別々にしょりしたほうがいいのでは

            # 得られた直線を元の画像に描画
            print(f"Total: {len(peak[0])}")
            num_of_horizontal_lines = 0
            num_of_vertical_lines = 0
            num_of_interdot_lines = 0
            lines_list = []
            all_lines_img = np.copy(rgb_img)
            for i in range(peak.shape[1]):
                rho = peak[0, i] * rho_res
                theta = peak[1, i] * theta_res - np.pi

                if np.sin(theta) != 0:
                    intercept = height - 1 - int(rho / np.sin(theta))
                    slope = -1 * np.cos(theta) / np.sin(theta)

                    one_line_image = np.copy(rgb_img)
                    if  slope < 0:
                        # interdot
                        num_of_interdot_lines += 1
                        lines_list.append(["interdot", -1 * slope, intercept * voltage_per_pixel, hough_array[peak[0, i], peak[1, i]]])
                        _line_rho_theta(one_line_image, rho, theta)
                        cv2.imwrite(output_folder + f"/individual_line/interdot_{i}.png", one_line_image)
                    elif slope < 1:
                        # horizontal
                        num_of_horizontal_lines += 1
                        lines_list.append(["horizontal", -1 * slope, intercept * voltage_per_pixel, hough_array[peak[0, i], peak[1, i]]])
                        _line_rho_theta(one_line_image, rho, theta)
                        cv2.imwrite(output_folder + f"/individual_line/horizontal_{i}.png", one_line_image)
                    else:
                        # vertical
                        num_of_vertical_lines += 1 
                        lines_list.append(["vertical", -1 * slope, intercept * voltage_per_pixel, hough_array[peak[0, i], peak[1, i]]])
                        _line_rho_theta(one_line_image, rho, theta)
                        cv2.imwrite(output_folder + f"/individual_line/vertical_{i}.png", one_line_image)
                else:
                    # 傾き無限大の場合
                    num_of_vertical_lines += 1
                    lines_list.append(["vertical", 'inf', rho, hough_array[peak[0, i], peak[1, i]]])
                    one_line_image = np.copy(rgb_img)
                    _line_rho_theta(one_line_image, rho, theta)
                    cv2.imwrite(output_folder + f"/individual_line/vertical_{i}.png", one_line_image)

                _line_rho_theta(all_lines_img, rho, theta)


            print(f"|- Vertical:   {num_of_vertical_lines}\n|- Interdot:   {num_of_interdot_lines}\n|- Horizontal: {num_of_horizontal_lines}")

            cv2.imwrite(output_folder + "/detected_lines.png", all_lines_img)
            df = pd.DataFrame(lines_list, columns=["type", "slope", "intercept", "votes"])
            df.sort_values(by="type").to_csv(output_folder + "/line_parapeters.csv", index=True)

        case "slope":
            # 目標: 傾きを３種類求める

            # 異なる閾値のもと、各領域ごとに傾きの最頻値をを取得
            theta_max_vertical = _calculate_theta_max(
                hough_array_vertical,
                theta_array_vertical, 
                threshold_vertical, 
                theta_res,
                #0,
            )
            theta_max_interdot = _calculate_theta_max(
                hough_array_interdot, 
                theta_array_interdot,
                threshold_interdot, 
                theta_res,
                #int(rng),
            )
            theta_max_horizontal = _calculate_theta_max(
                hough_array_horizontal,
                theta_array_horizontal, 
                threshold_horizontal, 
                theta_res,
                #int(rng * 3 / 2),
            )
            print("Detected Theta")
            print(f"|- vertical:    {int(theta_max_vertical*180/np.pi):4d} [degree]")
            print(f"|- interdot:    {int(theta_max_interdot*180/np.pi):4d} [degree]")
            print(f"|- horizontal:  {int(theta_max_horizontal*180/np.pi):4d} [degree]")

            lines_list = []
            lines_list.append(["vertical",   1 / np.tan(theta_max_vertical)    if np.tan(theta_max_vertical)!=0 else 0])
            lines_list.append(["interdot",   1 / np.tan(theta_max_interdot)    if np.tan(theta_max_interdot)!=0 else 0])
            lines_list.append(["horizontal", 1 / np.tan(theta_max_horizontal)  if np.tan(theta_max_horizontal)!=0 else 0])
            df = pd.DataFrame(lines_list, columns=["type", "slope"])
            df.sort_values(by="type").to_csv(output_folder + "/slope.csv", index=False)

            # 傾きを描画
            output_img = np.copy(rgb_img)
            _line_rho_theta(
                output_img, 
                int(width / 2 * np.cos(theta_max_vertical) + height / 2 * np.sin(theta_max_vertical)),
                theta_max_vertical
            )
            _line_rho_theta(
                output_img,
                int(width / 2 * np.cos(theta_max_interdot) + height / 2 * np.sin(theta_max_interdot)),
                theta_max_interdot
            )
            _line_rho_theta(
                output_img,
                int(width / 2 * np.cos(theta_max_horizontal) + height / 2 * np.sin(theta_max_horizontal)),
                theta_max_horizontal
            )
            cv2.imwrite(output_folder + "/detected_slope.png", output_img)

def generalized_hough_transform(
    
):
    pass


def _detect_peak_coordinate(
    hough_array: npt.NDArray,
    theta_array: npt.NDArray,
    threshold: int,
    #theta_gap: int,
) -> npt.NDArray:
    """
    出力：
        座標 [ [rho1, rho2, ...], [theta1, theta2, ...] ]
        ただし、
        theta の座標は np.arange(-rng, rng) の値 (-180...0...179) 
    """
    # 極大値を検出
    # cv2.HoughLinesを再現した以下のサイトを参考にした。
    # https://campkougaku.com/2021/08/17/hough3/#toc2
    thresholded_hough_array = np.copy(hough_array)
    thresholded_hough_array[thresholded_hough_array < threshold] = 0
 
    peak_local = (thresholded_hough_array[1:-1, 1:-1] >= thresholded_hough_array[0:-2, 1:-1]) * (thresholded_hough_array[1:-1, 1:-1] >= thresholded_hough_array[2:, 1:-1]) * \
           (thresholded_hough_array[1:-1, 1:-1] >= thresholded_hough_array[1:-1, 0:-2]) * (thresholded_hough_array[1:-1, 1:-1] >= thresholded_hough_array[1:-1, 2:]) * \
           (thresholded_hough_array[1:-1, 1:-1] > 0)
    peak_local = np.array(np.where(peak_local)) + 1  # peak: [ [rho1, rho2, ...], [theta1, theta2, ...] ]   
    peak_global = np.copy(peak_local)
    for i, t in enumerate(peak_local[1]):
        peak_global[1, i] = theta_array[t]
    #peak[1, :] += theta_gap
    print(peak_global)
    
    return peak_global

def _calculate_theta_max(
    hough_array: npt.NDArray,
    theta_array: npt.NDArray,
    threshold: int,
    theta_res: float,
    #theta_gap: int,
) -> float:
    """
    
    出力:
        rad
    """
    thresholded_hough_array = np.copy(hough_array)
    thresholded_hough_array[thresholded_hough_array < threshold] = 0
    thresholded_hough_array = np.sum(thresholded_hough_array, axis=0)

    local_index = np.argmax(thresholded_hough_array)
    global_index = theta_array[local_index]

    """
    # 確認用 (普段はコメントアウト)
    indices = np.arange(len(thresholded_hough_array))
    indices += theta_gap - 180
    plt.plot(indices, thresholded_hough_array, marker='o', linestyle='-')
    plt.title("Theta Histogram")
    plt.savefig(output_folder + f"/check_gap{theta_gap}.png")
    plt.close()
    """
    return global_index * theta_res - np.pi

def _line_rho_theta(
    img,
    rho,
    theta,
):
    """
    Args:
        img:
        rho: 
        theta: [rad]
    """
    height, width = img.shape[:2]
    x1 = 0
    x2 = width - 1
    if theta != 0 and theta != np.pi and theta != -np.pi:
        y1 = int(rho / np.sin(theta))
        y2 = int(y1 - x2 / np.tan(theta)) if np.cos(theta) != 0 else y1
        if theta < 0: 
            line_color = (0, 0, 255)
        elif 0 <= theta <= np.pi/2:
            line_color = (255, 0, 0)
        else:
            line_color = (0, 255, 0) 
        cv2.line(img, (x1, y1), (x2, y2), line_color, 1)
    else:
        cv2.line(img, (int(rho), 0), (int(rho), height-1), (0, 0, 255), 1)

def _split_hough_array(hough_array):
    hough_tmp = np.copy(hough_array)
    hough_tmp = np.split(hough_tmp, 8, axis=1)

    hough_array_m180_m90 = np.hstack((hough_tmp[0], hough_tmp[1]))         
    hough_array_m90_m45 =  hough_tmp[2]
    hough_array_m45_0 =    hough_tmp[3]
    hough_array_0_90 =     np.hstack((hough_tmp[4], hough_tmp[5]))
    hough_array_90_135 =   hough_tmp[6]               
    hough_array_135_180 =  hough_tmp[7]
    
    hough_array_vertical =   np.hstack((hough_array_m45_0, hough_array_135_180))
    hough_array_interdot =   np.hstack((hough_array_m180_m90, hough_array_0_90))
    hough_array_horizontal = np.hstack((hough_array_m90_m45, hough_array_90_135))

    return hough_array_vertical, hough_array_interdot, hough_array_horizontal

def _split_theta_array(theta_array):
    theta_tmp = np.copy(theta_array)
    theta_tmp = np.split(theta_tmp, 8)

    theta_array_m180_m90 = np.hstack((theta_tmp[0], theta_tmp[1]))         
    theta_array_m90_m45 =  theta_tmp[2]
    theta_array_m45_0 =    theta_tmp[3]
    theta_array_0_90 =     np.hstack((theta_tmp[4], theta_tmp[5]))
    theta_array_90_135 =   theta_tmp[6]               
    theta_array_135_180 =  theta_tmp[7]
    
    theta_array_vertical =   np.hstack((theta_array_m45_0, theta_array_135_180))
    theta_array_interdot =   np.hstack((theta_array_m180_m90, theta_array_0_90))
    theta_array_horizontal = np.hstack((theta_array_m90_m45, theta_array_90_135))

    return theta_array_vertical, theta_array_interdot, theta_array_horizontal




if __name__ == "__main__":

    """
    filepath_line = "./data/output_train_simu/result/6_class1.png"
    filepath_triangle = "./data/output_train_simu/result/6_class2.png"
    filepath = integrate_edges(filepath_line, filepath_triangle)
    
    filepath = "./data/output_utils/small.png"

    """
    filepath = "./data/_archive/canny.png"
    
    
    
    """
    detect_line_segment(
        filepath=filepath, 
        edge_extraction=True,
        hough_threshold=40,         
        hough_minLineLength=10,     
        hough_maxLineGap=5              
    )

    """
    hough_transform_CSD(
        method="slope_intercept",
        filepath=filepath,
        threshold_vertical=53,
        threshold_interdot=30,
        threshold_horizontal=30,
    )

    
    
