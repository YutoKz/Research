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

from model import UNet_2D
from utils import torch_fix_seed

torch_fix_seed()



dir_input = "./inputs/hitachi/raw/original/1.png"
#load_model = "./models/pretrain/pretrain_29.pth"
load_model = "./models/finetune/finetune_54.pth"
dir_output = "./outputs/infer"

# フォルダ準備
if os.path.exists(dir_output):
  shutil.rmtree(dir_output)

os.mkdir(dir_output) 

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

img_orig = cv2.imread(dir_input)

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

