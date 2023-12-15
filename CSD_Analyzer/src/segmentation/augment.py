import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
import cv2
import os
import shutil

def data_augmentation(
    dir_input: str,
    dir_output: str,
    seed: int = 42,
    num_augment: int = 5,
    RRC_size = (100, 100), 
    RRC_scale = (0.08, 1.0),
    RRC_ratio = (0.75, 1.333),

):
    """ データセットを拡張, pngで保存.
    Args:
        dir_input:  学習データセット
                    サブディレクトリに, label と original. 対応するファイルの名前は同じに.
        dir_output: 拡張した学習データセットの保存先
        seed: シード値

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
            transforms.PILToTensor(), # あくまで確認用
        ]
    )
    transform_label = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=RRC_size, scale=RRC_scale, ratio=RRC_ratio, interpolation=Image.NEAREST),
            transforms.PILToTensor(), # あくまで確認用
        ]
    )  # ラベル画像用に補間方法を変更



    # dir_input 内の各画像に
    for i, file_name in enumerate(os.listdir(dir_input + "/original")):
        print(file_name)
        # 画像読み込み
        original = Image.open(dir_input + "/original/" + file_name).convert('L')
        label = Image.open(dir_input + "/label/" + file_name).convert('L')

        # 元画像ペアにtransform_original / transform_label を 複数回 適用, 保存.
        ## original
        torch.manual_seed(seed)
        for j in range(num_augment):
            # 適用
            transformed_original = transform_original(original)      
            # pngで保存
            cv2.imwrite(dir_output + f"/original/{i*num_augment+j}.png", transformed_original.numpy()[0])
        ## label
        torch.manual_seed(seed)
        for j in range(num_augment):
            # 適用
            transformed_label = transform_label(label)
            # pngで保存
            cv2.imwrite(dir_output + f"/label/{i*num_augment+j}.png", transformed_label.numpy()[0])
            # 確認用!!!
            cv2.imwrite(dir_output + f"/check/{i*num_augment+j}.png",  transformed_label.numpy()[0]*100)



    
if __name__ == "__main__":
    data_augmentation(
        dir_input="./data/input_hitachi/before_augment",
        dir_output="./data/input_hitachi/augmented",
        
    )