import pandas as pd
"""
import cv2
from PIL import Image
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
"""

from sklearn.model_selection import train_test_split

data = pd.read_csv("/home/yuto_kizawa/Project/Research/UNet-try/imgpath_hw_labelpath.csv")

# データをシャッフル
shuffled_data = data.sample(frac=1, random_state=42)  # random_state はシャッフルの再現性を確保するためのもの

# データをトレーニング、検証、テストセットに分割 (例: 60%トレーニング, 20%検証, 20%テスト)
train_data, temp_data = train_test_split(shuffled_data, test_size=0.4)
validation_data, test_data = train_test_split(temp_data, test_size=0.5)

train_df = train_data.reset_index(drop=True)
val_df = validation_data.reset_index(drop=True)
test_df = test_data.reset_index(drop=True)

#print(len(data))
#print(len(train_df))
#print(len(val_df))
#print(len(test_df))
#print(train_df["imgpath"].str[-50:-20].head(5))


