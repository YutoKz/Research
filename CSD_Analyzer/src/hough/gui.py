import sys
import os
import numpy as np
import tkinter as tk

import tkinter.ttk as ttk
from PIL import Image, ImageTk
from tkinter import messagebox as tkMessageBox
from tkinter import filedialog as tkFileDialog
from tkinter import filedialog as filedialog

r_w = 300
r_h = 800

lt_w = 1000
lt_h = 400
lb_w = 1000
lb_h = 400

class Application(tk.Frame):
    DEBUG_LOG = True
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        
        self.input_filepath = ""
        
        self.method = "slope_intercept"
        self.edge_extraction = False
        self.lower_threshold = 0,
        self.upper_threshold = 10000000,
        self.lower_threshold_interdot = None,
        self.upper_threshold_interdot = None,
        self.voltage_per_pixel = 1.0,         # TODO: 縦横かえれるように
        self.rho_res = 0.5,
        self.theta_res = np.pi / 180,

        self.create_widgets()

    
    def create_widgets(self):
        # frame
        ## root
        fm_root = tk.Frame(root)
        fm_root.pack(fill=tk.BOTH, expand=False)
        ## root <- left / right
        fm_left          = tk.Frame(fm_root, bg="lightcoral", width=lt_w, height=lt_h+lb_h)
        fm_right         = tk.Frame(fm_root, bg="lightblue", width=r_w, height=r_h)
        fm_left.pack(side=tk.LEFT, fill=tk.NONE, expand=True)
        fm_right.pack(side=tk.RIGHT, fill=tk.NONE, expand=True)
        ## left <- left top / left bottom
        fm_left_top      = tk.Frame(fm_left, bg="lightgreen", width=lt_w, height=lt_h)
        fm_left_bottom   = tk.Frame(fm_left, bg="lightcoral", width=lb_w, height=lb_h)
        fm_left_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        fm_left_bottom.pack(side=tk.BOTTOM, fill=tk.NONE, expand=True)
        ## left top <- left top 0~2
        fm_left_top_0 =  tk.Frame(fm_left_top, bg="lightgreen")
        fm_left_top_1 =  tk.Frame(fm_left_top, bg="lightcoral")
        fm_left_top_2 =  tk.Frame(fm_left_top, bg="lightgreen")
        fm_left_top_0.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fm_left_top_1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fm_left_top_2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ## left bottom <- left bottom 0~2
        fm_left_bottom_0 =  tk.Frame(fm_left_bottom, bg="lightgreen")
        fm_left_bottom_1 =  tk.Frame(fm_left_bottom, bg="lightcoral")
        fm_left_bottom_2 =  tk.Frame(fm_left_bottom, bg="lightgreen")
        fm_left_bottom_0.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fm_left_bottom_1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fm_left_bottom_2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ## right <- right 0~2
        fm_right_0 =  tk.Frame(fm_right, bg="lightgreen")
        fm_right_1 =  tk.Frame(fm_right, bg="lightcoral")
        fm_right_2 =  tk.Frame(fm_right, bg="lightgreen")
        fm_right_0.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        fm_right_1.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        fm_right_2.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # left top
        ## label
        label_original = tk.Label(fm_left_top_0, text="Input")
        label_processed = tk.Label(fm_left_top_1, text="Processed")
        label_rhotheta = tk.Label(fm_left_top_2, text="rho - theta")
        label_original.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        label_processed.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        label_rhotheta.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ## image
        self.image_original = tk.Label(fm_left_top_0)
        self.image_processed = tk.Label(fm_left_top_1)
        self.image_rhotheta = tk.Label(fm_left_top_2)
        self.image_original.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.image_processed.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.image_rhotheta.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # left bottom
        ## widget
        label_h_i_v             = tk.Label      (fm_left_bottom_0, text="Horizontal / Interdot / Vertical")
        label_individual        = tk.Label      (fm_left_bottom_1, text="Individual Line")
        label_csv               = tk.Label      (fm_left_bottom_2, text="Line Parameter")
        self.image_horizontal   = tk.Label      (fm_left_bottom_0)
        self.image_interdot     = tk.Label      (fm_left_bottom_0)
        self.image_vertical     = tk.Label      (fm_left_bottom_0)
        self.image_individual   = tk.Label      (fm_left_bottom_1)
        self.tree_csv           = ttk.Treeview  (fm_left_bottom_2)
        ## pack
        label_h_i_v.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        label_individual.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        label_csv.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.image_horizontal.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.image_interdot.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.image_vertical.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.image_individual.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.tree_csv.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        ## tree csv
        ### show header
        self.tree_csv["show"] = "headings"
        self.tree_csv["columns"] = ("index","type","slope","intercept","votes")
        ### header width
        self.tree_csv.column("index",       width=10, anchor='e')
        self.tree_csv.column("type",        width=10, anchor='center')
        self.tree_csv.column("slope",       width=10, anchor='center')
        self.tree_csv.column("intercept",   width=10, anchor='center')
        self.tree_csv.column("votes",       width=10, anchor='center')
        ### header text
        self.tree_csv.heading("index",     text="index")
        self.tree_csv.heading("type",      text="type")
        self.tree_csv.heading("slope",     text="slope")
        self.tree_csv.heading("intercept", text="intercept")
        self.tree_csv.heading("votes",     text="votes")

        # right
        ## select input filepath
        browse_button = tk.Button(fm_right_0, text="Browse", command=self.browse_file)
        self.label_inputpath = tk.Label(fm_right_0, text="Select input filepath", width=30)
        browse_button.pack(side=tk.LEFT)
        self.label_inputpath.pack(side=tk.LEFT)
        ## change parameters of Hough
        ### widget
        label_method                    = tk.Label(fm_right_1, text="Method")
        label_edge_extraction           = tk.Label(fm_right_1, text="Edge extraction")
        label_thinning                  = tk.Label(fm_right_1, text="Thinning")
        label_lower_threshold           = tk.Label(fm_right_1, text="Lower threshold")
        label_upper_threshold           = tk.Label(fm_right_1, text="Upper threshold")
        label_lower_threshold_interdot  = tk.Label(fm_right_1, text="Lower threshold for Interdot")
        label_upper_threshold_interdot  = tk.Label(fm_right_1, text="Upper threshold for Interdot")
        label_voltage_per_pixel         = tk.Label(fm_right_1, text="V / px")
        options = ["Please select", "slope_intercept"]    # "slope" は抜いてある
        self.tkvar_method = tk.StringVar()
        self.tkvar_method.set(options[0])
        optionmenu_method = ttk.OptionMenu(fm_right_1, self.tkvar_method, *options, command=self.method_selected)
        self.tkvar_edge_extraction = tk.BooleanVar()
        checkbox_edge_extraction = tk.Checkbutton(fm_right_1, text="On / Off", variable=self.tkvar_edge_extraction, command=self.edge_extraction_checked)
        self.tkvar_thinning = tk.BooleanVar()
        checkbox_thinning = tk.Checkbutton(fm_right_1, text="On / Off", variable=self.tkvar_thinning, command=self.thinning_checked)
        self.spinbox_lower_threshold            = tk.Spinbox(fm_right_1, width=5, from_=0, to=100000, increment=1,       command=self.lower_threshold_changed)
        self.spinbox_upper_threshold            = tk.Spinbox(fm_right_1, width=5, from_=0, to=100000, increment=1,       command=self.upper_threshold_changed)
        self.spinbox_lower_threshold_interdot   = tk.Spinbox(fm_right_1, width=5, from_=0, to=100000, increment=1,       command=self.lower_threshold_interdot_changed)
        self.spinbox_upper_threshold_interdot   = tk.Spinbox(fm_right_1, width=5, from_=0, to=100000, increment=1,       command=self.upper_threshold_interdot_changed)
        self.spinbox_voltage_per_pixel          = tk.Spinbox(fm_right_1, width=5, from_=0.0, to=100000, increment=0.001, command=self.voltage_per_pixel_changed, format="%.3f")   
        ### grid
        label_method.grid                           (row=0, column=0, sticky=tk.E, padx=5, pady=15)
        label_edge_extraction.grid                  (row=1, column=0, sticky=tk.E, padx=5, pady=15)
        label_thinning.grid                         (row=2, column=0, sticky=tk.E, padx=5, pady=15)
        label_lower_threshold.grid                  (row=3, column=0, sticky=tk.E, padx=5, pady=15)
        label_upper_threshold.grid                  (row=4, column=0, sticky=tk.E, padx=5, pady=15)
        label_lower_threshold_interdot.grid         (row=5, column=0, sticky=tk.E, padx=5, pady=15)
        label_upper_threshold_interdot.grid         (row=6, column=0, sticky=tk.E, padx=5, pady=15)
        label_voltage_per_pixel.grid                (row=7, column=0, sticky=tk.E, padx=5, pady=15)
        optionmenu_method.grid                      (row=0, column=1, sticky=tk.W)       
        checkbox_edge_extraction.grid               (row=1, column=1, sticky=tk.W)
        checkbox_thinning.grid                      (row=2, column=1, sticky=tk.W)
        self.spinbox_lower_threshold.grid           (row=3, column=1, sticky=tk.W)
        self.spinbox_upper_threshold.grid           (row=4, column=1, sticky=tk.W)
        self.spinbox_lower_threshold_interdot.grid  (row=5, column=1, sticky=tk.W)
        self.spinbox_upper_threshold_interdot.grid  (row=6, column=1, sticky=tk.W)
        self.spinbox_voltage_per_pixel.grid         (row=7, column=1, sticky=tk.W)
        ### enter to change value
        self.spinbox_lower_threshold.bind('<Return>', self.lower_threshold_changed)
        self.spinbox_upper_threshold.bind('<Return>', self.upper_threshold_changed)       
        self.spinbox_lower_threshold_interdot.bind('<Return>', self.lower_threshold_interdot_changed)   
        self.spinbox_upper_threshold_interdot.bind('<Return>', self.upper_threshold_interdot_changed)   
        self.spinbox_voltage_per_pixel.bind('<Return>', self.voltage_per_pixel_changed)
        ### set initial value
        self.spinbox_lower_threshold.delete             (0, tk.END)         
        self.spinbox_upper_threshold.delete             (0, tk.END)         
        self.spinbox_lower_threshold_interdot.delete    (0, tk.END)
        self.spinbox_upper_threshold_interdot.delete    (0, tk.END)
        self.spinbox_voltage_per_pixel.delete           (0, tk.END)     
        self.spinbox_lower_threshold.insert             (0, "0")         
        self.spinbox_upper_threshold.insert             (0, "10000")         
        self.spinbox_lower_threshold_interdot.insert    (0, "0")
        self.spinbox_upper_threshold_interdot.insert    (0, "10000")
        self.spinbox_voltage_per_pixel.insert           (0, "1.0")    
        ## exec button
        ### widget
        button_exec = tk.Button(fm_right_2, text="Execute", command=        """hough_transform_CSD"""        )
        button_exec.pack(anchor="nw")





    # TODO: ここもっといろんな処理書かないと
    def browse_file(self):
        selected_filepath = filedialog.askopenfilename(filetypes=[("png", "*.png")])
        if selected_filepath:
            base_dir = os.getcwd()
            selected_filepath = "./" + os.path.relpath(selected_filepath, start=base_dir)
            self.label_inputpath.config(text=selected_filepath)
            self.input_filepath = selected_filepath


        else:
            self.label_inputpath.config(text="Select input filepath")

    def method_selected(self, selected_value):
        self.method = selected_value
        print(self.method)
    
    def edge_extraction_checked(self):
        self.edge_extraction = True if self.tkvar_edge_extraction.get() else False
        print(self.edge_extraction)

    def thinning_checked(self):
        self.thinning = True if self.tkvar_thinning.get() else False
        print(self.thinning)

    def lower_threshold_changed(self, event=None):
        self.lower_threshold = int(self.spinbox_lower_threshold.get())
        print(self.lower_threshold)
    
    def upper_threshold_changed(self, event=None):
        self.upper_threshold = int(self.spinbox_upper_threshold.get())
        print(self.upper_threshold)

    def lower_threshold_interdot_changed(self, event=None):
        self.lower_threshold_interdot = int(self.spinbox_lower_threshold_interdot.get())
        print(self.lower_threshold_interdot)

    def upper_threshold_interdot_changed(self, event=None):
        self.upper_threshold_interdot = int(self.spinbox_upper_threshold_interdot.get())
        print(self.upper_threshold_interdot)

    def voltage_per_pixel_changed(self, event=None):
        self.voltage_per_pixel = float(self.spinbox_voltage_per_pixel.get())
        print(self.voltage_per_pixel)
    

# 実行
root = tk.Tk()        
myapp = Application(master=root)
myapp.master.title("My Application") # タイトル
#myapp.master.geometry(f"{r_w+lt_w}x{r_h+lt_h}") # ウィンドウの幅と高さピクセル単位で指定（width x height）
#print(f"{r_w+lt_w}x{r_h}")

myapp.mainloop()