"""
    単一の画像を学習済みモデルに突っ込む
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
#from PIL import Image

import os

from model import UNet_2D

# GPU, Optimizer, Loss function
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = UNet_2D().to(device)

# test
model = UNet_2D()
model.load_state_dict(torch.load("./models/train_29.pth"))
#model.eval()
sigmoid = nn.Sigmoid()

img_orig = cv2.imread("./data/takahashi/big_csd_paper1_remove_text_gray.png")
#img_orig = cv2.imread("./data/csd.png")

required_height = [i for i in range(img_orig.shape[0] - 15, img_orig.shape[0] + 1) if i % 16 == 0]
required_width = [i for i in range(img_orig.shape[1] - 15, img_orig.shape[1] + 1) if i % 16 == 0]
print(f"height:{required_height}, width:{required_width}")
img = cv2.resize(img_orig, dsize = (required_width[0], required_height[0]))

# サイズを手動で変更したい場合に使用。普段はコメントアウト
img = cv2.resize(img_orig, dsize=(192, 192))

img = img/255
img = torch.from_numpy(img.astype(np.float32)).clone()
img = img.permute(2, 0, 1)
img = img.unsqueeze(0)

output = model(img)
output = sigmoid(output)
pred = torch.argmax(output, axis=1)
pred = torch.nn.functional.one_hot(pred.long(), num_classes=2).to(torch.float32)

orig = img[0,0,:,:].cpu().numpy()
cv2.imwrite("./data/infer_output/original.png", orig*255)

pred_np = pred[0,:,:,1].cpu().numpy()
cv2.imwrite("./data/infer_output/pred.png", pred_np*255)
