"""
  simuCSDで学習
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
from torcheval.metrics.functional import multiclass_accuracy
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
dir_input = "./data/input_simu"
dir_output = "./data/output_pretrain_before"

# make directories
if os.path.exists(dir_output):
  shutil.rmtree(dir_output)
#if os.path.exists("./models"):
#  shutil.rmtree("./models")

os.mkdir(dir_output)  
os.mkdir(dir_output+"/result")
#os.mkdir("./models")

# num of classes
classes = max(np.unique(np.array(cv2.imread(dir_input+"/label/0.png", cv2.IMREAD_GRAYSCALE))).tolist()) + 1
print(f"num of classes: {classes}")

# DataFrame
num_data = int(sum(os.path.isfile(os.path.join(dir_input+'/noisy', name)) for name in os.listdir(dir_input+'/noisy')) / 2)
num_data = 100 # データセットの一部を使いたい場合に使用。普段はコメントアウト
d = {"imgpath":[dir_input+f"/noisy/{i}_gray.png" for i in range(num_data)], "labelpath": [dir_input+f"/label/{i}.png" for i in range(num_data)]}
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

# Loss function
criterion = nn.CrossEntropyLoss()
print(f"Loss funtion: {str(criterion)[0:-2]}")
"""
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
def criterion(pred,target):
    return 0.5*BCELoss(pred, target) + 0.5*TverskyLoss(pred, target)

criterion = smp.losses.JaccardLoss(mode='multilabel')
"""



# Training
history = {"train_loss": [], "val_loss": []}
epochs = 30

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



# Loss
plt.figure()
plt.plot(history["train_loss"])
plt.xlabel('batch')
plt.ylabel('loss')
plt.savefig(dir_output+"/train_loss.png")
plt.figure()
plt.plot(history["val_loss"])
plt.xlabel('batch')
plt.ylabel('loss')
plt.savefig(dir_output+"/val_loss.png")






# test
#best_model_index = history["val_loss"].index(min(history["val_loss"]))
#print(f"Best Model: train_{best_model_index+1}.pth")
#os.rename(f"./models/train_{best_model_index+1}.pth", f"./models/train_{best_model_index+1}_best.pth")
model = UNet_2D(classes=classes)
model.load_state_dict(torch.load(f"./models/pretrained_{epochs}.pth"))
model.eval()
sigmoid = nn.Sigmoid()

with torch.no_grad():
  for i, data in enumerate(test_loader):
    inputs, labels = data["img"], data["label"]
    outputs = model(inputs)
    #loss = criterion(outputs, labels)
    #print("loss: ",loss.item())

    outputs = sigmoid(outputs)
    pred = torch.argmax(outputs, dim=1)
    pred = torch.nn.functional.one_hot(pred.long(), num_classes=classes).to(torch.float32)

    orig_np = inputs[0,0,:,:].cpu().numpy()
    cv2.imwrite(dir_output+f"/result/{i}_original.png", orig_np*255)

    #lab_np = labels[0,1,:,:].cpu().numpy()
    lab_np = torch.argmax(labels[0,:,:,:], dim=0).cpu().numpy()
    cv2.imwrite(dir_output+f"/result/{i}_label.png", lab_np*255//(classes-1))
    
    pred_np = torch.argmax(pred[0,:,:,:], dim=2).cpu().numpy()
    cv2.imwrite(dir_output+f"/result/{i}_pred.png", pred_np*255//(classes-1))
    for j in range(classes):
      if j != 0:
        pred_np = pred[0,:,:,j].cpu().numpy()
        cv2.imwrite(dir_output+f"/result/{i}_class{j}.png", pred_np*255)
