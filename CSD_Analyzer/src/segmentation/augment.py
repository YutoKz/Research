import torchvision.transforms as transforms
from PIL import Image
import torch

"""
seed = 42
torch.manual_seed(seed)

# 画像を読み込む
image_path = 'path_to_your_image.jpg'
label_path = 'path_to_your_label_image.jpg'

image = Image.open(image_path)
label = Image.open(label_path)

# 画像に対する変換を定義
transform_image = transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=Image.BILINEAR)
transform_image.seed = seed

# ラベル画像に対する変換を定義
transform_label = transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=Image.NEAREST)  # ラベル画像用に補間方法を変更
transform_label.seed = seed

# 同じ乱数シードを使用して変換を適用
transformed_image = transform_image(image)
transformed_label = transform_label(label)

"""

def data_augmentation(
    dir_input,
):
    """ データセットを拡張, pngで保存.
    Args:
        dir_input:  学習データセット
                    サブディレクトリに, label と original. ファイル名のフォーマットは 数字.png.
        dir_output: 拡張した学習データセットの保存先

    """
    # transform をoriginal / label 個別に定義
    # dir_input 内の各画像に
        # 画像読み込み

        # transformのseedを42に固定

        # dir_output に元画像ペア保存　<- ここたぶん不要や

        # 元画像ペアにtransform_original / transform_label を 複数回 適用, 保存.
            # 1回分
            # 適用
    
            # pngで保存. ファイル名は 数値.png 
    
