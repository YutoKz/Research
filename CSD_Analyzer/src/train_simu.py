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
import os
import shutil

from dataset import Dataset
from model import UNet_2D

# make directories
if os.path.exists("./data/train_simu_output"):
  shutil.rmtree("./data/train_simu_output")
if os.path.exists("./models"):
  shutil.rmtree("./models")

os.mkdir("./data/train_simu_output")  
os.mkdir("./data/train_simu_output/result")
os.mkdir("./models")

# DataFrame
num_data = int(sum(os.path.isfile(os.path.join('./data/simu_input/noisy', name)) for name in os.listdir('./data/simu_input/noisy')) / 2)
d = {"imgpath":[f"./data/simu_input/noisy/{i}_gray.png" for i in range(num_data)], "labelpath": [f"./data/simu_input/original/{i}.png" for i in range(num_data)]}
data = pd.DataFrame(data=d)

# shuffle data
shuffled_data = data.sample(frac=1, random_state=42)  # random_state はシャッフルの再現性を確保するためのもの

# train:validation:test = 3:1:1
train_data, temp_data = train_test_split(shuffled_data, test_size=0.4, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_df = train_data.reset_index(drop=True)
val_df = validation_data.reset_index(drop=True)
test_df = test_data.reset_index(drop=True)

# DataLoader
BATCH_SIZE = 8
train_dataset = Dataset(train_df)
val_dataset = Dataset(val_df)
test_dataset = Dataset(test_df)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0,shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)


# GPU, Optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = UNet_2D().to(device)
optimizer = optim.Adam(unet.parameters(), lr=0.001)

# Loss function
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
def criterion(pred,target):
    return 0.5*BCELoss(pred, target) + 0.5*TverskyLoss(pred, target)
#criterion = nn.CrossEntropyLoss()


# Training
history = {"train_loss": [], "val_loss": []}
n = 0
m = 0
epochs = 5

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
    if i % ((len(train_df)//BATCH_SIZE)//10) == (len(train_df)//BATCH_SIZE)//10 - 1:   # df ⇒ train_dfに修正した
      print(f"epoch:{epoch+1}  index:{i+1}  train_loss:{train_loss/n:.5f}")
      n = 0
      train_loss = 0
      train_acc = 0


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
plt.savefig("./data/train_simu_output/train_loss.png")
plt.figure()
plt.plot(history["val_loss"])
plt.xlabel('batch')
plt.ylabel('loss')
plt.savefig("./data/train_simu_output/val_loss.png")

# test
best_model_index = history["val_loss"].index(min(history["val_loss"]))
model = UNet_2D()
model.load_state_dict(torch.load(f"./models/train_{best_model_index+1}.pth"))
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
    pred = torch.nn.functional.one_hot(pred.long(), num_classes=2).to(torch.float32)

    orig = inputs[0,0,:,:].cpu().numpy()
    cv2.imwrite(f"./data/train_simu_output/result/{i}_original.png", orig*255)

    lab = labels[0,1,:,:].cpu().numpy()
    cv2.imwrite(f"./data/train_simu_output/result/{i}_label.png", lab*255)
    
    pred_np = pred[0,:,:,1].cpu().numpy()
    cv2.imwrite(f"./data/train_simu_output/result/{i}_pred.png", pred_np*255)
