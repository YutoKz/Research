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
from sklearn.model_selection import train_test_split
import os
import shutil

from dataset import Dataset
from model import UNet_2D

# 学習データセットのフォルダ名
folder_name = "input_simu"

# make directories
if os.path.exists("./data/output_train_simu"):
  shutil.rmtree("./data/output_train_simu")
if os.path.exists("./models"):
  shutil.rmtree("./models")

os.mkdir("./data/output_train_simu")  
os.mkdir("./data/output_train_simu/result")
os.mkdir("./models")

# DataFrame
num_data = int(sum(os.path.isfile(os.path.join('./data/'+folder_name+'/noisy', name)) for name in os.listdir('./data/'+folder_name+'/noisy')) / 2)

# クラス数
classes = max(np.unique(np.array(cv2.imread("./data/"+folder_name+"/label/0.png", cv2.IMREAD_GRAYSCALE))).tolist()) + 1
print(f"num of classes: {classes}")

# データセットの一部を使いたい場合に使用。普段はコメントアウト
num_data = 300

d = {"imgpath":[f"./data/{folder_name}/noisy/{i}_gray.png" for i in range(num_data)], "labelpath": [f"./data/{folder_name}/label/{i}.png" for i in range(num_data)]}
data = pd.DataFrame(data=d)

# shuffle data
shuffled_data = data.sample(frac=1, random_state=42)  # random_state はシャッフルの再現性を確保するためのもの

# train:validation:test = 3:1:1
train_data, temp_data = train_test_split(shuffled_data, test_size=0.4, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_df = train_data.reset_index(drop=True)
val_df = validation_data.reset_index(drop=True)
test_df = test_data.reset_index(drop=True)

print(f"num of training data: {len(train_df)}")
print()

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
"""
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
def criterion(pred,target):
    return 0.5*BCELoss(pred, target) + 0.5*TverskyLoss(pred, target)
"""
criterion = nn.CrossEntropyLoss()


# Training
history = {"train_loss": [], "val_loss": []}
n = 0
m = 0
epochs = 10

for epoch in range(epochs):
  train_loss = 0
  val_loss = 0

  unet.train()
  for i, data in enumerate(train_loader):
    inputs, labels = data["img"].to(device), data["label"].to(device)
    optimizer.zero_grad()
    outputs = unet(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    history["train_loss"].append(loss.item())
    n += 1
    """ データ数減らす実験のため一旦コメントアウト
    if i % ((len(train_df)//BATCH_SIZE)//10) == (len(train_df)//BATCH_SIZE)//10 - 1:   # df ⇒ train_dfに修正した
      print(f"epoch:{epoch+1}  index:{i+1}  train_loss:{train_loss/n:.5f}")
      n = 0
      train_loss = 0
      train_acc = 0
    """

  unet.eval()
  with torch.no_grad():
    for i, data in enumerate(val_loader):
      inputs, labels = data["img"].to(device), data["label"].to(device)
      outputs = unet(inputs)
      loss = criterion(outputs, labels)
      val_loss += loss.item()
      m += 1
      if i % (len(val_df)//BATCH_SIZE) == len(val_df)//BATCH_SIZE - 1:
        print(f"epoch:{epoch+1}  index:{i+1}  val_loss:{val_loss/m:.5f}")
        print()
        history["val_loss"].append(val_loss/m)
        m = 0
        val_loss = 0
        val_acc = 0
  torch.save(unet.state_dict(), f"./models/train_{epoch+1}.pth")
print("finish training")


# Loss
plt.figure()
plt.plot(history["train_loss"])
plt.xlabel('batch')
plt.ylabel('loss')
plt.savefig("./data/output_train_simu/train_loss.png")
plt.figure()
plt.plot(history["val_loss"])
plt.xlabel('batch')
plt.ylabel('loss')
plt.savefig("./data/output_train_simu/val_loss.png")

# test
best_model_index = history["val_loss"].index(min(history["val_loss"]))
print(f"Best Model: train_{best_model_index+1}.pth")
os.rename(f"./models/train_{best_model_index+1}.pth", f"./models/train_{best_model_index+1}_best.pth")
model = UNet_2D(classes=classes)
model.load_state_dict(torch.load(f"./models/train_{best_model_index+1}_best.pth"))
model.eval()
sigmoid = nn.Sigmoid()

with torch.no_grad():
  for i, data in enumerate(test_loader):
    inputs, labels = data["img"], data["label"]
    outputs = model(inputs)
    #loss = criterion(outputs, labels)
    #print("loss: ",loss.item())

    outputs = sigmoid(outputs)
    pred = torch.argmax(outputs, axis=1)
    pred = torch.nn.functional.one_hot(pred.long(), num_classes=classes).to(torch.float32)

    orig_np = inputs[0,0,:,:].cpu().numpy()
    cv2.imwrite(f"./data/output_train_simu/result/{i}_original.png", orig_np*255)

    #lab_np = labels[0,1,:,:].cpu().numpy()
    lab_np = torch.argmax(labels[0,:,:,:], dim=0).cpu().numpy()
    cv2.imwrite(f"./data/output_train_simu/result/{i}_label.png", lab_np*255//(classes-1))
    
    for j in range(classes):
      if j != 0:
        pred_np = pred[0,:,:,j].cpu().numpy()
        cv2.imwrite(f"./data/output_train_simu/result/{i}_class{j}.png", pred_np*255)
