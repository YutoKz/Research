"""
pytorch_lightning を使ったすっきりした実装

どうもテストがうまくいってなさそう

当面は普通にPyTorch単体での使用がよさそう。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
     
# データの読み込み（df: data frame）
df = pd.read_csv('wine_class.csv')
    
# データの表示（先頭の5件）
print(df.head())

# 正解ラベル数確認
# np.unique(df['Class'], return_counts=True)

# 入力変数と正解ラベルへ分離
x = df.drop('Class', axis=1)
t = df['Class']

# pandasの形式からTensorへ直接の変換不可
# .valuesで一旦numpyの形式へ
x = torch.tensor(x.values, dtype=torch.float32)
t = torch.tensor(t.values, dtype=torch.int64)

# ラベルを 0 から始める
t = t - 1       

# 入力変数と目的変数をまとめて、ひとつのオブジェクト dataset に変換
dataset = torch.utils.data.TensorDataset(x, t)

# 各データセットのサンプル数を決定
# train : val : test = 60% : 20% : 20%
n_train = int(len(dataset) * 0.6)
n_val = int((len(dataset) - n_train) * 0.5)
n_test = len(dataset) - n_train - n_val

# ランダムに分割を行うため、シードを固定して再現性を確保
torch.manual_seed(0)

# データセットの分割
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

# DataLoader は Lightning が自動で設定


#-------------------------------------------------------------------------------------
# TrainNet, Vali..., TestNetを継承したNetクラスを状況に応じ編集する

# 学習データに対する処理
class TrainNet(pl.LightningModule):

    # @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(train, self.batch_size, shuffle=True)

    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'loss': loss}
        return results

      
# 検証データに対する処理
class ValidationNet(pl.LightningModule):

    # @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(val, self.batch_size,     num_workers=16)

    def validation_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        results = {'val_loss': loss, 'val_acc': acc}
        return results

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        results = {'val_loss': avg_loss, 'val_acc': avg_acc}

        print(results[val_loss])

        return results

      
# テストデータに対する処理
class TestNet(pl.LightningModule):

    # @pl.data_loader
    def test_dataloader(self):
        return torch.utils.data.DataLoader(test, self.batch_size,     num_workers=16)

    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        results = {'test_loss': loss, 'test_acc': acc}
        return results

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        results = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return results

      
# 学習データ、検証データ、テストデータへの処理を継承したクラス
class Net(TrainNet, ValidationNet, TestNet):
    
    def __init__(self, input_size=10, hidden_size=5, output_size=3, batch_size=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def lossfun(self, y, t):
        return F.cross_entropy(y, t)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

# 学習に関する一連の流れを実行
torch.manual_seed(0)

net = Net()
trainer = Trainer(max_epochs=100)

trainer.fit(net)

# テストデータに対する処理の実行（test_step と test_end）
trainer.test()

#  テストデータに対する結果の確認
print(trainer.callback_metrics)