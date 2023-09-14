"""
おそらく問題なく実行できた
"""

from sklearn.datasets import load_iris
import torch 
import torch.nn as nn
import torch.nn.functional as F

#-------------------------------------------------------------------------------------------

# Iris データセット読み込み、Tensorへ変換
x, t = load_iris(return_X_y=True)
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)

# 入力変数と目的変数をまとめて、ひとつのオブジェクト dataset に変換
dataset = torch.utils.data.TensorDataset(x, t)

# 各データセットのサンプル数を決定
# train : val: test = 60%　: 20% : 20%
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - n_train - n_val

# ランダムに分割を行うため、シードを固定して再現性を確保
torch.manual_seed(0)

# データセットの分割
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

#-------------------------------------------------------------------------------------------

# バッチサイズ
batch_size = 10
      
# shuffle はデフォルトで False のため、学習データのみ True に指定
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)

#-------------------------------------------------------------------------------------------

class Net(nn.Module):

    # 使用するオブジェクトを定義
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 3)

    # 順伝播
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

#-------------------------------------------------------------------------------------------

# 演算に使用できる GPU の有無を確認
print('cuda is available:', torch.cuda.is_available())

# GPU の設定状況に基づいたデバイスの選択
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# エポックの数
max_epoch = 1

# 乱数のシードを固定して再現性を確保
torch.manual_seed(0)

# モデルのインスタンス化とデバイスへの転送
net = Net().to(device)

# 目的関数の設定
criterion = F.cross_entropy
  
# 最適化手法の選択
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)


# 学習ループ
for epoch in range(max_epoch):
    
    for batch in train_loader:
        
        # バッチサイズ分のサンプルを抽出
        x, t = batch
        
        # 学習時に使用するデバイスへデータの転送
        x = x.to(device)
        t = t.to(device)
        
        # パラメータの勾配を初期化
        optimizer.zero_grad()
        
        # 予測値の算出
        y = net(x)
        
        # 目標値と予測値から目的関数の値を算出
        loss = criterion(y, t)
        
        # 目的関数の値を表示して確認
        # item(): tensot.Tensor => float
        print('loss: ', loss.item())
        

        # dim=1 で行ごとの最大値に対する要素番号を取得（dim=0 は列ごと）
        y_label = torch.argmax(y, dim=1)
        
        # 正解率
        acc  = torch.sum(y_label == t) * 1.0 / len(t)
        print('accuracy:', acc)

        # 各パラメータの勾配を算出
        loss.backward()
        
        # 勾配の情報を用いたパラメータの更新
        optimizer.step()


# 正解率の計算
def calc_acc(data_loader):
    
    with torch.no_grad():
        
        accs = [] # 各バッチごとの結果格納用
        
        for batch in data_loader:
            x, t = batch
            x = x.to(device)
            t = t.to(device)
            y = net(x)
            
            y_label = torch.argmax(y, dim=1)
            acc = torch.sum(y_label == t) * 1.0 / len(t)
            accs.append(acc)
            
    # 全体の平均を算出
    avg_acc = torch.tensor(accs).mean()
    print('Accuracy: {:.1f}%'.format(avg_acc * 100))
    
    return avg_acc


      
# 検証データで確認
calc_acc(val_loader)

# テストデータで確認
calc_acc(test_loader)



