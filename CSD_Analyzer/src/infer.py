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
model.load_state_dict(torch.load("./models/train_5.pth"))
#model.eval()
sigmoid = nn.Sigmoid()

img = cv2.imread("./data/takahashi/csd_paper1_remove_text_gray.png")
#img = cv2.resize(img, dsize = (144, 144))   #応急処置　元々2のべき乗とかならいらん
required_height = [i for i in range(img.shape[0] - 15, img.shape[0] + 1) if i % 16 == 0]
required_width = [i for i in range(img.shape[1] - 15, img.shape[1] + 1) if i % 16 == 0]
print(f"{required_height},{required_width}")
img = cv2.resize(img, dsize = (required_width[0], required_height[0]))
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
