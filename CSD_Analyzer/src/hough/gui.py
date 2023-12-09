import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import csv

class LineDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Line Detection GUI")

        # メインフレーム
        self.main_frame = ttk.Frame(root)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 変数
        self.image_path = None
        self.detected_lines = []

        # ウィジェットの作成
        self.create_widgets()

    def create_widgets(self):
        # ファイル選択ボタン
        file_btn = ttk.Button(self.main_frame, text="画像を選択", command=self.load_image)
        file_btn.grid(row=0, column=0, columnspan=2, pady=10)

        # ハフ変換パラメータの設定
        ttk.Label(self.main_frame, text="パラメータ設定").grid(row=1, column=0, columnspan=2, pady=5)

        ttk.Label(self.main_frame, text="Threshold:").grid(row=2, column=0, sticky=tk.E, pady=5)
        self.threshold_entry = ttk.Entry(self.main_frame)
        self.threshold_entry.grid(row=2, column=1, pady=5)
        self.threshold_entry.insert(0, "50")

        # 実行ボタン
        run_btn = ttk.Button(self.main_frame, text="直線検出実行", command=self.run_line_detection)
        run_btn.grid(row=3, column=0, columnspan=2, pady=10)

        # 結果表示エリア
        ttk.Label(self.main_frame, text="結果表示").grid(row=4, column=0, columnspan=2, pady=5)

        self.image_canvas = tk.Canvas(self.main_frame, width=400, height=400)
        self.image_canvas.grid(row=5, column=0, columnspan=2, pady=5)

        # 検出結果表示ボタン
        show_result_btn = ttk.Button(self.main_frame, text="検出結果表示", command=self.show_detection_result)
        show_result_btn.grid(row=6, column=0, pady=5)

        # CSV表示エリア
        ttk.Label(self.main_frame, text="CSV表示").grid(row=7, column=0, columnspan=2, pady=5)

        self.csv_text = tk.Text(self.main_frame, width=40, height=10)
        self.csv_text.grid(row=8, column=0, columnspan=2, pady=5)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(title="画像を選択", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])

    def run_line_detection(self):
        if self.image_path is None:
            print("画像が選択されていません。")
            return

        # ハフ変換パラメータの取得
        threshold = int(self.threshold_entry.get())

        # 画像読み込み
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # ハフ変換
        lines = cv2.HoughLines(image, 1, np.pi / 180, threshold)

        # 検出結果の保存
        self.detected_lines = lines

        # 検出結果を画像に描画
        image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if lines is not None:
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
                cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 画像をキャンバスに表示
        self.display_image(image_with_lines)

        # CSVに検出結果を保存
        self.save_lines_to_csv()




    def show_detection_result(self):
        if len(self.detected_lines) == 0:
            print("検出された直線がありません。")
            return

        # 画像読み込み
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)

        # 検出結果を画像に描画
        image_with_lines = image.copy()
        for line in self.detected_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 画像をキャンバスに表示
        self.display_image(image_with_lines)

    def save_lines_to_csv(self):
        if len(self.detected_lines) == 0:
            print("検出された直線がありません。")
            return

        # CSVに検出結果を保存
        csv_data = [["rho", "theta"]]
        for line in self.detected_lines:
            rho, theta = line[0]
            csv_data.append([rho, theta])

        csv_file_path = "detected_lines.csv"
        with open(csv_file_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_data)

        # CSVをテキストエリアに表示
        with open(csv_file_path, "r") as csv_file:
            csv_text = csv_file.read()
            self.csv_text.delete(1.0, tk.END)
            self.csv_text.insert(tk.END, csv_text)

    def display_image(self, image):
        # 画像をリサイズ
        height, width = image.shape[:2]
        ratio = 400 / max(height, width)
        new_height = int(height * ratio)
        new_width = int(width * ratio)
        resized_image = cv2.resize(image, (new_width, new_height))

        # OpenCVの画像をTkinter PhotoImageに変換
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        photo = tk.PhotoImage(width=new_width, height=new_height)
        photo.put("{R G B} " + " ".join(map(str, image_rgb.flatten())))

        # キャンバスに画像を表示
        self.image_canvas.config(width=new_width, height=new_height)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.image_canvas.photo = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = LineDetectionApp(root)
    root.mainloop()
