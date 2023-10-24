"""
    単一の画像を学習済みモデルに突っ込む
"""
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import functional
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
#from PIL import Image

import os


# UNet
class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size = 3, padding="same")
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.rl = nn.ReLU()
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size = 3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x

class UNet_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 64, 64)
        self.TCB2 = TwoConvBlock(64, 128, 128)
        self.TCB3 = TwoConvBlock(128, 256, 256)
        self.TCB4 = TwoConvBlock(256, 512, 512)
        self.TCB5 = TwoConvBlock(512, 1024, 1024)
        self.TCB6 = TwoConvBlock(1024, 512, 512)
        self.TCB7 = TwoConvBlock(512, 256, 256)
        self.TCB8 = TwoConvBlock(256, 128, 128)
        self.TCB9 = TwoConvBlock(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4= UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 2, kernel_size = 1) #変更　64, 4 -> 64, 2
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        #print(x.shape)
        x = self.TCB1(x)
        #print(x.shape)
        x1 = x
        x = self.maxpool(x)
        #print(x.shape)

        x = self.TCB2(x)
        #print(x.shape)
        x2 = x
        x = self.maxpool(x)
        #print(x.shape)

        x = self.TCB3(x)
        #print(x.shape)
        x3 = x
        x = self.maxpool(x)
        #print(x.shape)

        x = self.TCB4(x)
        #print(x.shape)
        x4 = x
        x = self.maxpool(x)
        #print(x.shape)

        x = self.TCB5(x)
        #print(x.shape)

        x = self.UC1(x)
        #print(x.shape)
        x = torch.cat([x4, x], dim = 1)
        #print(x.shape)
        x = self.TCB6(x)
        #print(x.shape)

        x = self.UC2(x)
        #print(x.shape)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return x

# GPU, Optimizer, Loss function
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = UNet_2D().to(device)

# test
model = UNet_2D()
model.load_state_dict(torch.load("./output/models/train_15.pth"))
#model.eval()
sigmoid = nn.Sigmoid()

img = cv2.imread("./big_csd_paper1_remove_text_gray.png")
img = cv2.resize(img, dsize = (96, 96))   #応急処置　元々2のべき乗とかならいらん
img = img/255
img = torch.from_numpy(img.astype(np.float32)).clone()
img = img.permute(2, 0, 1)
img = img.unsqueeze(0)

output = model(img)
output = sigmoid(output)
pred = torch.argmax(output, axis=1)
pred = torch.nn.functional.one_hot(pred.long(), num_classes=2).to(torch.float32)

orig = img[0,0,:,:].cpu().numpy()
cv2.imwrite("./original.png", orig*255)

pred_np = pred[0,:,:,1].cpu().numpy()
cv2.imwrite("./pred.png", pred_np*255)


"""
with torch.no_grad():
  for i, data in enumerate(test_loader):
    inputs, labels = data["img"], data["label"]
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    #print("loss: ",loss.item())

    outputs = sigmoid(outputs)
    pred = torch.argmax(outputs, axis=1)
    pred = torch.nn.functional.one_hot(pred.long(), num_classes=2).to(torch.float32)

    orig = inputs[0,0,:,:].cpu().numpy()
    cv2.imwrite(f"./output/result/{i}_original.png", orig*255)

    lab = labels[0,1,:,:].cpu().numpy()
    cv2.imwrite(f"./output/result/{i}_label.png", lab*255)
    
    pred_np = pred[0,:,:,1].cpu().numpy()
    cv2.imwrite(f"./output/result/{i}_pred.png", pred_np*255)

"""