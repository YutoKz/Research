"""
  実験CSDで再学習

  TODO: 学習の流れを、pretrain/finetune をmethodとして関数にまとめてもいいかも
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
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import os
import shutil

from dataset import Dataset
from model import UNet_2D

# fix seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# directiory name
dir_input = "./data/input_finetune"
dir_output = "./data/output_finetune"

# make directories
if os.path.exists(dir_output):
  shutil.rmtree(dir_output)
os.mkdir(dir_output)  
os.mkdir(dir_output+"/result")

# num of classes
classes = max(np.unique(np.array(cv2.imread(dir_input+"/label/0.png", cv2.IMREAD_GRAYSCALE))).tolist()) + 1
print(f"num of classes: {classes}")

# DataFrame
num_data = int(sum(os.path.isfile(os.path.join(dir_input+'/original', name)) for name in os.listdir(dir_input+'/original')) / 2)
d = {"imgpath":[dir_input+f"/original/{i}_gray.png" for i in range(num_data)], "labelpath": [dir_input+f"/label/{i}.png" for i in range(num_data)]}
data = pd.DataFrame(data=d)

## shuffle data
shuffled_data = data.sample(frac=1, random_state=42)  # random_state はシャッフルの再現性を確保するためのもの

## train:validation:test = 3:1:1
train_data, temp_data = train_test_split(shuffled_data, test_size=0.4, random_state=seed)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)

train_df = train_data.reset_index(drop=True)
val_df = validation_data.reset_index(drop=True)
test_df = test_data.reset_index(drop=True)

print(f"num of training data: {len(train_df)}")

# DataLoader
BATCH_SIZE = 8
train_dataset = Dataset(train_df, classes=classes)
val_dataset = Dataset(val_df, classes=classes)
test_dataset = Dataset(test_df, classes=classes)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0,shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

# GPU, Optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = UNet_2D(classes=classes).to(device)
optimizer = optim.Adam(unet.parameters(), lr=0.001)




#　TODO: IOUが最大のモデルを使うため、ここで最適なモデルを判定
unet.load_state_dict(torch.load('pretrained.pth'))




# Loss function
criterion = nn.CrossEntropyLoss()
print(f"Loss funtion: {str(criterion)[0:-2]}")






# Fine-tuning
history = {"train_loss": [], "val_loss": []}
epochs = 10
for epoch in range(epochs):
  unet.train()
  print("-----------------------------------------")
  print(f"epoch: {epoch+1}")
  print("[training]")
  for i, data in enumerate(train_loader):
    inputs, labels = data["img"].to(device), data["label"].to(device)  # [データ数, クラス数, 縦, 横]
    optimizer.zero_grad()
    outputs = unet(inputs)

    loss = criterion(outputs, labels)
    
    loss.backward()
    optimizer.step()

    print(f"| - batch: {i+1}  loss: {loss.item():.5f}")
    history["train_loss"].append(loss.item())

  unet.eval()
  sigmoid = nn.Sigmoid()
  print("[validation]")
  with torch.no_grad():
    for i, data in enumerate(val_loader):
      inputs, labels = data["img"].to(device), data["label"].to(device)   # いずれも [データ数, クラス数, 縦, 横]
      outputs = unet(inputs)                                              # [データ数, クラス数, 縦, 横]

      # 性能評価
      ## Loss
      loss = criterion(outputs, labels)  
      history["val_loss"].append(loss.item())
      ## Accuracy, IOU
      outputs = sigmoid(outputs)
      pred = torch.argmax(outputs, dim=1)   # [データ数, 縦, 横]
      target = torch.argmax(labels, dim=1)  # [データ数, 縦, 横]
      tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels.to(torch.int), mode='multilabel', threshold=0.5)

      batch_accuracy  = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
      batch_f1        = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
      batch_iou       = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

      print(f"| - batch: {i+1}\n\t| - loss:      {loss.item():.5f}")
      print(f"\t| - Accuracy:  {batch_accuracy.item():.5f}")
      print(f"\t| - F1 score:  {batch_f1.item():.5f}")
      print(f"\t| - IOU score: {batch_iou.item():.5f}")

  print()
  torch.save(unet.state_dict(), f"./models/pretrained_{epoch+1}.pth")
print("finish training")