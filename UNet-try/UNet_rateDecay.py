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


# データフレーム
#   dockerの場合変更の必要あり
data = pd.read_csv("/usr/UNet-try/path_hw_docker.csv")

#   データをシャッフル
shuffled_data = data.sample(frac=1, random_state=42)  # random_state はシャッフルの再現性を確保するためのもの

#   データをトレーニング、検証、テストセットに分割 (例: 60%トレーニング, 20%検証, 20%テスト)
train_data, temp_data = train_test_split(shuffled_data, test_size=0.4, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_df = train_data.reset_index(drop=True)
val_df = validation_data.reset_index(drop=True)
test_df = test_data.reset_index(drop=True)


# Dataset, DataLoader
class Dataset(BaseDataset):
  def __init__(
      self,
      df,
      transform = None,
      classes = None,
      augmentation = None
      ):
    self.imgpath_list = df.imgpath
    self.labelpath_list = df.labelpath

  def __getitem__(self, i):
    imgpath = self.imgpath_list[i]
    img = cv2.imread(imgpath)
    img = cv2.resize(img, dsize = (256, 256))
    img = img/255
    img = torch.from_numpy(img.astype(np.float32)).clone()
    img = img.permute(2, 0, 1)

    labelpath = self.labelpath_list[i]
    #label = Image.open(labelpath)  ここを変えたことで、labelの次元が2次元から3次元になった可能性
    #label = np.asarray(label)
    label = cv2.imread(labelpath, cv2.IMREAD_GRAYSCALE)
    label = np.array(label)
    label = cv2.resize(label, dsize = (256, 256))
    label = torch.from_numpy(label.astype(np.float32)).clone()
    label = torch.nn.functional.one_hot(label.long(), num_classes=4)
    label = label.to(torch.float32)
    label = label.permute(2, 0, 1) # (2, 0, 1) ⇒ (0, 3, 1, 2)

    data = {"img": img, "label": label}
    return data
  
  def __len__(self):
    return len(self.imgpath_list)

BATCH_SIZE = 8

train_dataset = Dataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0,shuffle=True)

val_dataset = Dataset(val_df)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)

test_dataset = Dataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)


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

        self.conv1 = nn.Conv2d(64, 4, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
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
optimizer = optim.Adam(unet.parameters(), lr=0.001)

TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
def criterion(pred,target):
    return 0.5*BCELoss(pred, target) + 0.5*TverskyLoss(pred, target)


# Training
history = {"train_loss": []}
n = 0
m = 0

for epoch in range(30):
  # 学習率減衰
  if epoch > 10 and epoch <= 20:
    new_learning_rate = 0.0001  
    for param_group in optimizer.param_groups:
      param_group['lr'] = new_learning_rate
  elif epoch > 20:
    new_learning_rate = 0.00001  
    for param_group in optimizer.param_groups:
      param_group['lr'] = new_learning_rate
  
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
        m = 0
        val_loss = 0
        val_acc = 0

  torch.save(unet.state_dict(), f"./models/train_{epoch+1}.pth")
print("finish training")


# Loss
plt.plot(history["train_loss"])
plt.xlabel('batch')
plt.ylabel('loss')
plt.savefig("/usr/UNet-try/result_png/train_loss.png")


# test
model = UNet_2D()
model.load_state_dict(torch.load("./models/train_30.pth"))
model.eval()
with torch.no_grad():
  data = next(iter(test_loader))
  inputs, labels = data["img"], data["label"]
  outputs = model(inputs)
  loss = criterion(outputs, labels)
  print("loss: ",loss.item())

sigmoid = nn.Sigmoid()
outputs = sigmoid(outputs)
pred = torch.argmax(outputs, axis=1)
pred = torch.nn.functional.one_hot(pred.long(), num_classes=4).to(torch.float32)


# Result
plt.figure()
plt.imshow(data["img"][0,:,:,:].permute(1, 2, 0))
plt.title("original_image")
plt.axis("off")
plt.savefig("/usr/UNet-try/result_png/original.png")

plt.figure()
classes = ["background","large_bowel","small_bowel","stomach"]
fig, ax = plt.subplots(2, 4, figsize=(15,8))
for i in range(2):
  for j, cl in enumerate(classes):
    if i == 0:
      ax[i,j].imshow(pred[0,:,:,j])
      ax[i,j].set_title(f"pred_{cl}")
      ax[i,j].axis("off")
    else:
      ax[i,j].imshow(data["label"][0,j,:,:])    
      ax[i,j].set_title(f"label_{cl}")
      ax[i,j].axis("off")
plt.savefig("/usr/UNet-try/result_png/pred.png")
