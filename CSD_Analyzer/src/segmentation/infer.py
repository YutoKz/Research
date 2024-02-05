"""
    単一の画像を学習済みモデルに突っ込む
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
#from PIL import Image

import os
import shutil
import sys

from model import UNet_2D
from utils import torch_fix_seed, integrate_edges
import segmentation_models_pytorch as smp

torch_fix_seed()


arguments = sys.argv


input_img = "./inputs/hitachi/raw/original/1.png"
input_label = "./inputs/hitachi/raw/label/1_label_012.png"

if arguments[1] == "pretrain":
  print("--- mode: pretrain ---")
  load_model = "./models/pretrain/pretrain_22.pth"
elif arguments[1] == "finetune":
  print("--- mode: finetune ---")
  load_model = "./models/finetune/finetune_1.pth"
else:
   load_model = ""

dir_output = "./outputs/infer"

# フォルダ準備
if os.path.exists(dir_output):
  shutil.rmtree(dir_output)

os.mkdir(dir_output) 

shutil.copy(load_model, dir_output)

# 分類するクラス数
classes = 3

# GPU, Optimizer, Loss function
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = UNet_2D(classes=classes).to(device)

# test
model = UNet_2D(classes=classes)
model.load_state_dict(torch.load(load_model))

#model.eval()
sigmoid = nn.Sigmoid()

img_orig = cv2.imread(input_img)

required_height = [i for i in range(img_orig.shape[0] - 15, img_orig.shape[0] + 1) if i % 16 == 0]
required_width = [i for i in range(img_orig.shape[1] - 15, img_orig.shape[1] + 1) if i % 16 == 0]
print(f"Output size: ({required_height[0]}, {required_width[0]})")
img = cv2.resize(img_orig, dsize = (required_width[0], required_height[0]))

# サイズを手動で変更したい場合に使用。普段はコメントアウト
#img = cv2.resize(img_orig, dsize=(192, 192))

img = img/255
img = torch.from_numpy(img.astype(np.float32)).clone()
img = img.permute(2, 0, 1)
img = img.unsqueeze(0)

output = model(img)
output = sigmoid(output)
pred = torch.argmax(output, dim=1)
pred = torch.nn.functional.one_hot(pred.long(), num_classes=classes).to(torch.float32)

orig = img[0,0,:,:].cpu().numpy()
cv2.imwrite(dir_output + "/original.png", orig*255)

pred_np = torch.argmax(pred[0,:,:,:], dim=2).cpu().numpy()
cv2.imwrite(dir_output + f"/pred.png", pred_np*255//(classes-1))
cv2.imwrite(dir_output + f"/pred_binary.png", np.where(pred_np != 0, 1, pred_np)*255)
for j in range(classes):
      if j != 0:
        pred_np = pred[0,:,:,j].cpu().numpy()
        cv2.imwrite(dir_output + f"/pred_class{j}.png", pred_np*255)

integrate_edges(
   dir_output+"/pred_class1.png",
   dir_output+"/pred_class2.png",
   dir_output+"/integrated_edge.png"
)




label = cv2.imread(input_label, cv2.IMREAD_GRAYSCALE)
label = np.array(label)
label = cv2.resize(label, dsize=(required_width[0], required_height[0]))
label = torch.from_numpy(label.astype(np.float32)).clone()
label = torch.nn.functional.one_hot(label.long(), num_classes=classes)
label = label.to(torch.float32)
label = label.permute(2, 0, 1)
label = label.unsqueeze(0)
pred = pred.permute(0, 3, 1, 2)

tp, fp, fn, tn = smp.metrics.get_stats(pred, label.to(torch.int), mode='multilabel', threshold=0.5)
iou_micro = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
iou_class = smp.metrics.iou_score(tp.sum(0), fp.sum(0), fn.sum(0), tn.sum(0), reduction="none")
print("TP/FP/FN/TN")
print("| - tp: " + str(tp.sum(0).numpy()))
print("| - fp: " + str(fp.sum(0).numpy()))
print("| - fn: " + str(fn.sum(0).numpy()))
print("| - tn: " + str(tn.sum(0).numpy()))
print("Micro IoU")
print(f"| - micro: {iou_micro.item():.5f}")
print("Class IoU")
for c in range(classes):
  print(f"| - class {c}: {iou_class.tolist()[c]:.5f}")