import sys
import os
import numpy as np
import tkinter as tk

import tkinter.ttk as ttk
from PIL import Image, ImageTk
from tkinter import messagebox as tkMessageBox
from tkinter import filedialog as tkFileDialog

r_w = 300
r_h = 1000

lt_w = 1000
lt_h = 500
lb_w = 1000
lb_h = 500

class Application(tk.Frame):
    DEBUG_LOG = True
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()

        self.create_widgets()

    
    def create_widgets(self):
        # frame
        ## root
        fm_root = tk.Frame(root)
        fm_root.pack(fill=tk.BOTH, expand=True)
        ## root <- left / right
        fm_right         = tk.Frame(fm_root, bg="lightblue", width=r_w, height=r_h)
        fm_left          = tk.Frame(fm_root, bg="lightcoral", width=lt_w, height=lt_h+lb_h)
        fm_right.pack(side=tk.RIGHT, fill=tk.NONE, expand=True)
        fm_left.pack(side=tk.LEFT, fill=tk.NONE, expand=True)
        ## left <- left top / left bottom
        fm_left_top      = tk.Frame(fm_left, bg="lightgreen", width=lt_w, height=lt_h)
        fm_left_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        fm_left_bottom   = tk.Frame(fm_left, bg="lightcoral", width=lb_w, height=lb_h)
        fm_left_bottom.pack(side=tk.BOTTOM, fill=tk.NONE, expand=True)
        ## 

        # left top
        ## label
        label_original = tk.Label(fm_left_top, text="Input", width=20)
        label_original.grid(row=0, column=0)
        label_processed = tk.Label(fm_left_top, text="Processed")
        label_processed.grid(row=0, column=1)
        label_rhotheta = tk.Label(fm_left_top, text="rho - theta")
        label_rhotheta.grid(row=0, column=2)
        ## image
        self.panel_img = tk.Label(fm_left_top)
        self.panel_img.grid(row=1, column=0)



# 実行
root = tk.Tk()        
myapp = Application(master=root)
myapp.master.title("My Application") # タイトル
#myapp.master.geometry("1600x1000") # ウィンドウの幅と高さピクセル単位で指定（width x height）

myapp.mainloop()