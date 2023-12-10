import sys
import os
import numpy as np
import cv2
import tkinter as tk

import tkinter.ttk as ttk
from PIL import Image, ImageTk
from tkinter import messagebox as tkMessageBox
from tkinter import filedialog as tkFileDialog
from tkinter import filedialog as filedialog
from tkinter import scrolledtext

from hough import hough_transform, hough_transform_CSD

output_folder = "./data/output_hough"


r_w = 300
r_h = 600

lt_w = 1000
lt_h = 300
lb_w = 1000
lb_h = 300

class Application(tk.Frame):
    DEBUG_LOG = True
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()


        # hough parameter
        self.method = "slope_intercept"
        self.input_filepath = ""
        self.edge_extraction = False
        self.thinning = False
        self.lower_threshold = 0
        self.upper_threshold = 100000
        self.use_threshold_interdot = False
        self.lower_threshold_interdot = 0
        self.upper_threshold_interdot = 100000
        self.voltage_per_pixel = 1.0         # TODO: 縦横かえれるように
        self.rho_res = 0.5
        self.theta_res : float = np.pi / 180.0


        self.create_widgets()


    
    def create_widgets(self):
        # frame
        ## root
        self.fm_root             = tk.Frame(root)
        ## root <- left / right
        self.fm_left             = tk.Frame(self.fm_root, bg="lightgreen", padx=5)
        self.fm_right            = tk.Frame(self.fm_root, bg="lightblue", padx=5)
        ## left <- left top / left bottom
        self.fm_left_top         = tk.Frame(self.fm_left, bg="lightgreen", padx=5)
        self.fm_left_bottom      = tk.Frame(self.fm_left, bg="lightcoral", padx=5)
        ## left top <- left top 0~2
        self.fm_left_top_0       =  tk.Frame(self.fm_left_top, bg="lightgreen", padx=5)
        self.fm_left_top_1       =  tk.Frame(self.fm_left_top, bg="lightcoral", padx=5)
        self.fm_left_top_2       =  tk.Frame(self.fm_left_top, bg="lightgreen", padx=5)
        ## left bottom <- left bottom 0~2
        self.fm_left_bottom_0    =  tk.Frame(self.fm_left_bottom, bg="lightgreen", padx=5)
        self.fm_left_bottom_1    =  tk.Frame(self.fm_left_bottom, bg="lightcoral", padx=5)
        self.fm_left_bottom_2    =  tk.Frame(self.fm_left_bottom, bg="lightgreen", padx=5)
        ## right <- right 0~2
        self.fm_right_0          =  tk.Frame(self.fm_right, bg="lightblue", padx=5)
        self.fm_right_1          =  tk.Frame(self.fm_right, bg="lightblue", padx=5)
        self.fm_right_2          =  tk.Frame(self.fm_right, bg="lightblue", padx=5)
        
        # widget
        ## left top
        self.label_original          = tk.Label(self.fm_left_top_0, text="Input")
        self.label_processed         = tk.Label(self.fm_left_top_1, text="Processed")
        self.label_rhotheta          = tk.Label(self.fm_left_top_2, text="rho - theta")
        self.label_image_original     = tk.Label(self.fm_left_top_0)
        self.label_image_processed    = tk.Label(self.fm_left_top_1)
        self.label_image_rhotheta     = tk.Label(self.fm_left_top_2)
        ## left bottom
        self.label_h_i_v             = tk.Label      (self.fm_left_bottom_0, text="Horizontal / Interdot / Vertical")
        self.label_individual        = tk.Label      (self.fm_left_bottom_1, text="Individual Line")
        self.label_csv               = tk.Label      (self.fm_left_bottom_2, text="Line Parameter")
        self.label_image_horizontal   = tk.Label      (self.fm_left_bottom_0)
        self.label_image_interdot     = tk.Label      (self.fm_left_bottom_0)
        self.label_image_vertical     = tk.Label      (self.fm_left_bottom_0)
        self.label_image_individual   = tk.Label      (self.fm_left_bottom_1)
        self.tree_csv           = ttk.Treeview  (self.fm_left_bottom_2)
        ## right
        ### select input filepath
        self.browse_button                   = tk.Button(self.fm_right_0, text="Browse", command=self.browse_file)
        self.label_inputpath                 = tk.Label(self.fm_right_0, text="Select input filepath", width=30)
        ### change parameters of Hough
        self.label_method                    = tk.Label(self.fm_right_1, text="Method")
        self.label_edge_extraction           = tk.Label(self.fm_right_1, text="Edge extraction")
        self.label_thinning                  = tk.Label(self.fm_right_1, text="Thinning")
        self.label_lower_threshold           = tk.Label(self.fm_right_1, text="Lower threshold")
        self.label_upper_threshold           = tk.Label(self.fm_right_1, text="Upper threshold")
        self.label_threshold_interdot        = tk.Label(self.fm_right_1, text="Use unique threshold for Interdot line") 
        self.label_lower_threshold_interdot  = tk.Label(self.fm_right_1, text="Lower threshold for Interdot")
        self.label_upper_threshold_interdot  = tk.Label(self.fm_right_1, text="Upper threshold for Interdot")
        self.label_voltage_per_pixel         = tk.Label(self.fm_right_1, text="V / px")
        options = [" ", "slope_intercept"]    # "slope" は抜いてある
        self.tkvar_method           = tk.StringVar()
        self.tkvar_method.set(options[0])
        self.optionmenu_method           = ttk.OptionMenu(self.fm_right_1, self.tkvar_method, *options, command=self.method_selected)
        self.tkvar_edge_extraction  = tk.BooleanVar()
        self.checkbox_edge_extraction    = tk.Checkbutton(self.fm_right_1, text="Do?", variable=self.tkvar_edge_extraction, command=self.edge_extraction_checked)
        self.tkvar_thinning         = tk.BooleanVar()
        self.checkbox_thinning                  = tk.Checkbutton(self.fm_right_1, text="Do?", variable=self.tkvar_thinning, command=self.thinning_checked)
        self.spinbox_lower_threshold            = tk.Spinbox(self.fm_right_1, width=5, from_=0, to=10000, increment=1,       command=self.lower_threshold_changed)
        self.spinbox_upper_threshold            = tk.Spinbox(self.fm_right_1, width=5, from_=0, to=10000, increment=1,       command=self.upper_threshold_changed)
        self.tkvar_thereshold_interdot          = tk.BooleanVar()
        self.checkbox_threshold_interdot        = tk.Checkbutton(self.fm_right_1, text="Use?",  variable=self.tkvar_thereshold_interdot, command=self.threshold_interdot_checked)
        self.spinbox_lower_threshold_interdot   = tk.Spinbox(self.fm_right_1, width=5, from_=0, to=10000, increment=1,       command=self.lower_threshold_interdot_changed)
        self.spinbox_upper_threshold_interdot   = tk.Spinbox(self.fm_right_1, width=5, from_=0, to=10000, increment=1,       command=self.upper_threshold_interdot_changed)
        self.spinbox_voltage_per_pixel          = tk.Spinbox(self.fm_right_1, width=5, from_=0.0, to=10000, increment=0.001, command=self.voltage_per_pixel_changed, format="%.3f")   
        ### exec button
        self.button_exec         = tk.Button(self.fm_right_2, text="Execute", command=        """hough_transform_CSD"""        )
        self.scrolledtext_output = scrolledtext.ScrolledText(self.fm_right_2, wrap=tk.WORD, width=40, height=10)
        
        # pack / grid
        self.pack_grid()

        # configure
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
        ## spinbox
        ### enter to change value
        self.spinbox_lower_threshold.bind           ('<Return>', self.lower_threshold_changed)
        self.spinbox_upper_threshold.bind           ('<Return>', self.upper_threshold_changed)       
        self.spinbox_lower_threshold_interdot.bind  ('<Return>', self.lower_threshold_interdot_changed)   
        self.spinbox_upper_threshold_interdot.bind  ('<Return>', self.upper_threshold_interdot_changed)   
        self.spinbox_voltage_per_pixel.bind         ('<Return>', self.voltage_per_pixel_changed)
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
        ## scrolledtext
        self.scrolledtext_output.insert(tk.END, "Num of Detected Lines\n| - \n| - \n| -")
        self.scrolledtext_output.configure(state=tk.DISABLED)




    

    def pack_grid(self):
        ## root 
        self.fm_root.pack           (fill=tk.BOTH,                  expand=False)
        self.fm_left.pack           (side=tk.LEFT, fill=tk.NONE,    expand=True)
        self.fm_right.pack          (side=tk.RIGHT, fill=tk.NONE, expand=True)
        self.fm_left_top.pack       (side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fm_left_bottom.pack    (side=tk.BOTTOM, fill=tk.NONE, expand=True)
        self.fm_left_top_0.pack     (side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fm_left_top_1.pack     (side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fm_left_top_2.pack     (side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fm_left_bottom_0.pack  (side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fm_left_bottom_1.pack  (side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fm_left_bottom_2.pack  (side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fm_right_0.pack        (side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fm_right_1.pack        (side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fm_right_2.pack        (side=tk.TOP, fill=tk.BOTH, expand=True)
        ## left top
        self.label_original.pack    (side=tk.TOP,anchor=tk.N, fill=tk.BOTH, expand=False)
        self.label_processed.pack   (side=tk.TOP, fill=tk.BOTH, expand=False)
        self.label_rhotheta.pack    (side=tk.TOP, fill=tk.BOTH, expand=False)
        self.label_image_original.pack    (side=tk.BOTTOM, fill=tk.BOTH, expand=False)
        self.label_image_processed.pack   (side=tk.BOTTOM, fill=tk.BOTH, expand=False)
        self.label_image_rhotheta.pack    (side=tk.BOTTOM, fill=tk.BOTH, expand=False)
        ## left bottom
        self.label_h_i_v.pack       (side=tk.TOP, fill=tk.BOTH, expand=True)
        self.label_individual.pack  (side=tk.TOP, fill=tk.BOTH, expand=True)
        self.label_csv.pack         (side=tk.TOP, fill=tk.BOTH, expand=True)
        self.label_image_horizontal.pack  (side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.label_image_interdot.pack    (side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.label_image_vertical.pack    (side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.label_image_individual.pack  (side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.tree_csv.pack          (side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        ## right
        self.browse_button.pack                     (side=tk.LEFT)
        self.label_inputpath.pack                   (side=tk.LEFT)
        self.label_method.grid                      (row=0, column=0, sticky=tk.E, padx=5, pady=10)
        self.label_edge_extraction.grid             (row=1, column=0, sticky=tk.E, padx=5, pady=10)
        self.label_thinning.grid                    (row=2, column=0, sticky=tk.E, padx=5, pady=10)
        self.label_lower_threshold.grid             (row=3, column=0, sticky=tk.E, padx=5, pady=10)
        self.label_upper_threshold.grid             (row=4, column=0, sticky=tk.E, padx=5, pady=10)
        self.label_threshold_interdot.grid          (row=5, column=0,  sticky=tk.E, padx=5, pady=10)
        self.label_lower_threshold_interdot.grid    (row=6, column=0, sticky=tk.E, padx=5, pady=10)
        self.label_upper_threshold_interdot.grid    (row=7, column=0, sticky=tk.E, padx=5, pady=10)
        self.label_voltage_per_pixel.grid           (row=8, column=0, sticky=tk.E, padx=5, pady=10)
        self.optionmenu_method.grid                 (row=0, column=1, sticky=tk.W)       
        self.checkbox_edge_extraction.grid          (row=1, column=1, sticky=tk.W)
        self.checkbox_thinning.grid                 (row=2, column=1, sticky=tk.W)
        self.spinbox_lower_threshold.grid           (row=3, column=1, sticky=tk.W)
        self.spinbox_upper_threshold.grid           (row=4, column=1, sticky=tk.W)
        self.checkbox_threshold_interdot.grid       (row=5, column=1, sticky=tk.W)
        self.spinbox_lower_threshold_interdot.grid  (row=6, column=1, sticky=tk.W)
        self.spinbox_upper_threshold_interdot.grid  (row=7, column=1, sticky=tk.W)
        self.spinbox_voltage_per_pixel.grid         (row=8, column=1, sticky=tk.W)
        self.button_exec.pack                       (anchor="center", fill=tk.X, expand=True, padx=30, pady=15)
        self.scrolledtext_output.pack               (anchor="ne", fill=tk.X, expand=True)

    def browse_file(self):
        selected_filepath = filedialog.askopenfilename(filetypes=[("png", "*.png")])
        if selected_filepath:
            base_dir = os.getcwd()
            selected_filepath = "./" + os.path.relpath(selected_filepath, start=base_dir)

            # show/set filepath
            self.label_inputpath.config(text=selected_filepath)
            self.input_filepath = selected_filepath
            
            # show/set original and rhotheta
            self.original_image = self.resized_image(selected_filepath)
            tmp = cv2.imread(selected_filepath, cv2.IMREAD_GRAYSCALE)
            hough_array = hough_transform(tmp, self.rho_res, self.theta_res)
            self.rhotheta_image = self.resized_image(output_folder+"/rho_theta.png")
            self.label_image_original.configure(image=self.original_image)
            self.label_image_processed.configure(image=self.original_image)
            self.label_image_rhotheta.configure(image=self.rhotheta_image)
            self.pack_grid()
            
            # hough threshold
            vote_max = np.max(hough_array)
            vote_min = np.min(hough_array)
            ## variable
            self.lower_threshold = self.lower_threshold_interdot = vote_min
            self.upper_threshold = self.upper_threshold_interdot = vote_max
            ## spinbox
            ### change from_/to
            self.spinbox_lower_threshold.configure              (from_=vote_min, to=vote_max)
            self.spinbox_lower_threshold_interdot.configure     (from_=vote_min, to=vote_max)
            self.spinbox_upper_threshold.configure              (from_=vote_min, to=vote_max)
            self.spinbox_upper_threshold_interdot.configure     (from_=vote_min, to=vote_max)
            ### initialize 
            self.spinbox_lower_threshold.delete             (0, tk.END)         
            self.spinbox_upper_threshold.delete             (0, tk.END)         
            self.spinbox_lower_threshold_interdot.delete    (0, tk.END)
            self.spinbox_upper_threshold_interdot.delete    (0, tk.END)   
            self.spinbox_lower_threshold.insert             (0, str(vote_min))         
            self.spinbox_upper_threshold.insert             (0, str(vote_max))         
            self.spinbox_lower_threshold_interdot.insert    (0, str(vote_min))
            self.spinbox_upper_threshold_interdot.insert    (0, str(vote_max))

            print(self.lower_threshold)
            print(self.upper_threshold)

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

    def threshold_interdot_checked(self):
        self.use_threshold_interdot = True if self.tkvar_thereshold_interdot.get() else False
        print(self.use_threshold_interdot)

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

    def execute_pressed(self):
        hough_transform_CSD(
            method=self.method,
            filepath=self.input_filepath,
            edge_extraction=self.edge_extraction,
            thinning=self.thinning,
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
            lower_threshold_interdot=self.lower_threshold_interdot if self.use_threshold_interdot else None,
            upper_threshold_interdot=self.upper_threshold_interdot if self.use_threshold_interdot else None,
            voltage_per_pixel=self.voltage_per_pixel,
            rho_res=self.rhores,
            theta_res=self.thota_res,
        )



    def resized_image(self, filepath):
        image = Image.open(filepath)
        w, h = image.size
        image = image.resize((int(lt_h * w / h), lt_h))
        return ImageTk.PhotoImage(image)
    

# 実行
root = tk.Tk()        
myapp = Application(master=root)
myapp.master.title("My Application") # タイトル
#myapp.master.geometry(f"{r_w+lt_w}x{r_h+lt_h}") # ウィンドウの幅と高さピクセル単位で指定（width x height）
#print(f"{r_w+lt_w}x{r_h}")

myapp.mainloop()