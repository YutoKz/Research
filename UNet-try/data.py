import pandas as pd

"""

元画像path, ラベルpath を要素とするcsv 「path.csv」 を作成

/home/yuto_kizawa/Project/Research/UNet-CSD/train/case123/case123_day20/scans/
slice_0065_266_266_1.50_1.50.png

case123_day20_slice_0065    ←この3つのindexで一意に指定可能

"""

df = pd.read_csv("/home/yuto_kizawa/Project/Research/UNet-CSD/with_label.csv")

# imgpath labelpath



