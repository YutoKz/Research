import cv2
import numpy as np

# 画像の読み込み
image = cv2.imread("./data/output_infer/class1.png", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("./data/output_hough/original.png", image)

# エッジ検出
#edges = cv2.Canny(image, 50, 100)
#cv2.imwrite("./data/output_hough/canny.png", edges)

# ハフ変換による直線検出
#lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=30)
lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=8)
print(lines[0])

# imageをRGBへ
rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# 各直線に対する処理
for line in lines:
    """
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # 直線を描画
    cv2.line(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    """

    x1, y1, x2, y2 = line[0]

    # 直線を描画
    cv2.line(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # 直線の切片と傾きを計算
    slope = (y2 - y1) / (x2 - x1 + 1e-6)  # ゼロ除算を防ぐために小さな値を追加
    intercept = y1 - slope * x1

    print(f"直線の切片: {intercept}, 傾き: {slope}")

# 結果の保存
cv2.imwrite("./data/output_hough/detected_lines.png", rgb_image)
