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
    required_height = [i for i in range(img.shape[0] - 15, img.shape[0] + 1) if i % 16 == 0]   # 解像度が16の倍数となるよう調整
    required_width = [i for i in range(img.shape[1] - 15, img.shape[1] + 1) if i % 16 == 0]   # 解像度が16の倍数となるよう調整
    img = cv2.resize(img, dsize = (required_width[0], required_height[0]))
    img = img/255
    img = torch.from_numpy(img.astype(np.float32)).clone()
    img = img.permute(2, 0, 1)

    labelpath = self.labelpath_list[i]
    #label = Image.open(labelpath)  ここを変えたことで、labelの次元が2次元から3次元になった可能性
    #label = np.asarray(label)
    label = cv2.imread(labelpath, cv2.IMREAD_GRAYSCALE)
    label = np.array(label)
    required_height = [i for i in range(label.shape[0] - 15, label.shape[0] + 1) if i % 16 == 0]   # 解像度が16の倍数となるよう調整
    required_width = [i for i in range(label.shape[1] - 15, label.shape[1] + 1) if i % 16 == 0]   # 解像度が16の倍数となるよう調整
    label = cv2.resize(label, dsize = (required_width[0], required_height[0]))    
    label = torch.from_numpy(label.astype(np.float32)).clone()
    label = torch.nn.functional.one_hot(label.long(), num_classes=2)
    label = label.to(torch.float32)
    label = label.permute(2, 0, 1) # (2, 0, 1) ⇒ (0, 3, 1, 2)

    data = {"img": img, "label": label}
    return data
  
  def __len__(self):
    return len(self.imgpath_list)
