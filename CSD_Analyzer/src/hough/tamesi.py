import tkinter as tk
from tkinter import filedialog

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if file_path:
        # ファイルが選択された場合、パスをラベルに表示
        label.config(text=f"Selected File: {file_path}")
    else:
        # キャンセルされた場合の処理
        label.config(text="No file selected")

# GUIウィンドウの作成
root = tk.Tk()
root.title("File Browser")

# ボタンとラベルの作成
browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.pack(pady=10)

label = tk.Label(root, text="Selected File: None")
label.pack(pady=10)

# イベントループの開始
root.mainloop()
