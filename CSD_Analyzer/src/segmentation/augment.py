import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
import cv2
import os
import shutil
from utils import torch_fix_seed

# fix seed
torch_fix_seed()

def data_augmentation(
    dir_input: str,
    dir_output: str,
    seed: int = 42,
    augmentation_ratio: int = 5,
    RRC_size = (100, 100), 
    RRC_scale = (0.3, 1.0),
    RRC_ratio = (0.75, 1.333),
    RE_prob: float = 0.0,

):
    """ データセットを拡張, pngで保存.
    Args:
        dir_input:  学習データセット
                    サブディレクトリに, label と original. 対応するファイルの名前は同じに.
        dir_output: 拡張した学習データセットの保存先
        seed: シード値

        augmentation_ratio: 1枚の画像から何枚生成するか

        RRC_        RandomResizedCrop のパラメータ
            size:   切り抜くサイズ(出力のサイズ)
            scale:  リサイズする倍率
            ratio:  変化後のアスペクト比


    """
    if os.path.exists(dir_output):
        shutil.rmtree(dir_output)
    os.mkdir(dir_output)
    os.mkdir(dir_output + "/label")  
    os.mkdir(dir_output + "/original")  
    os.mkdir(dir_output + "/check")  

    # transform をoriginal / label 個別に定義
    transform_original = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=RRC_size, scale=RRC_scale, ratio=RRC_ratio, interpolation=Image.BILINEAR),
            transforms.PILToTensor(),
            transforms.RandomErasing(p=RE_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ]
    )
    transform_label = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=RRC_size, scale=RRC_scale, ratio=RRC_ratio, interpolation=Image.NEAREST),    # ラベル画像用に補間方法を変更
            transforms.PILToTensor(),
            transforms.RandomErasing(p=RE_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ]
    )  



    # dir_input 内の各画像に
    for i, file_name in enumerate(os.listdir(dir_input + "/original")):
        print(file_name)
        # 画像読み込み
        original = Image.open(dir_input + "/original/" + file_name).convert('L')
        label = Image.open(dir_input + "/label/" + file_name).convert('L')

        # 元画像ペアにtransform_original / transform_label を 複数回 適用, 保存.
        ## original
        torch.manual_seed(seed)
        for j in range(augmentation_ratio):
            # 適用
            transformed_original = transform_original(original)   
            # pngで保存
            cv2.imwrite(dir_output + f"/original/{i*augmentation_ratio+j}.png", transformed_original.numpy()[0])
        ## label
        torch.manual_seed(seed)
        for j in range(augmentation_ratio):
            # 適用
            transformed_label = transform_label(label)
            # pngで保存
            cv2.imwrite(dir_output + f"/label/{i*augmentation_ratio+j}.png", transformed_label.numpy()[0])
            # 確認用!!!
            cv2.imwrite(dir_output + f"/check/{i*augmentation_ratio+j}.png",  transformed_label.numpy()[0]*100)



    
if __name__ == "__main__":
    data_augmentation(
        dir_input="./inputs/hitachi/dataset1217",
        dir_output="./outputs/augment",
        augmentation_ratio=3,
        RRC_size=(64, 64),
        RRC_scale=(0.5, 0.7), #(0.3, 0.5),
        RRC_ratio=(0.75, 1.333),
        RE_prob=0.0,
    )