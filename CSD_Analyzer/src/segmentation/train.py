"""
    モデルを選択して学習
    simuCSDで事前学習 / realCSDでFine-tuning
"""
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
import segmentation_models_pytorch as smp
#from torcheval.metrics.functional import multiclass_accuracy
from sklearn.model_selection import train_test_split
import os
import shutil
import sys
import math
from typing import Literal
from dataset import Dataset
from model import UNet_2D
from utils import integrate_edges, torch_fix_seed

MethodType = Literal["pretrain", "finetune"]
LossType = Literal["CrossEntropyLoss", "JaccardLoss", "DiceLoss"]

# fix seed
torch_fix_seed()

def train(
    method: MethodType,
    dir_input,
    dir_output,
    classes,
    device, 
    model: nn.Module,
    loaded_model_index: int = None,
    num_data: int = None,
    val_percent: float = 0.1,
    test_percent: float = 0.1,
    loss_type: LossType = "CrossEntropyLoss",
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    early_stopping: bool = True,
    patience: int = 5,
) -> int: 
    """
    Args:
        method: 事前学習 / Fine-tuning
        dir_input:  学習用データを格納
                    サブディレクトリに original と label
                    フォーマット: original/0.png label/0.png
        dir_output: 出力を格納
        classes:    分類クラス数
        device:     ex) device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model:      ex) unet = UNet_2D(classes=classes).to(device)
        loaded_model_index: Fine-tuningする場合にセットする事前学習モデルのindex
        num_data:       学習用データの一部を使う場合に使用
        val_percent:    学習用データ内のvalidation dataの割合
        test_percent:   学習用データ内のtest dataの割合
        epochs:
        batch_size: 
        learning_rate: 
        patience: 学習を早期終了する基準epoch数。patience以上精度が悪化し続けたら終了。

    Returns:
        validationデータで最もIOUが高かった学習モデルの番号。
    """
    print(f"\n--- mode: {method} ---")
    print(f"device: {device}")
    print(f"model:  {type(model)}")
    print(f"early stopping: {early_stopping}")
    if early_stopping:
        print(f"| - patience: {patience}")


    # fix seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # make directories
    if os.path.exists(dir_output):
        shutil.rmtree(dir_output)
    os.mkdir(dir_output)  
    os.mkdir(dir_output+"/result")

    if method == "pretrain":
        if os.path.exists("./models"):
            shutil.rmtree("./models")
        os.mkdir("./models")
        os.mkdir("./models/pretrain")
        os.mkdir("./models/finetune")

    # num of classes
    #classes = max(np.unique(np.array(cv2.imread(dir_input+"/label/0.png", cv2.IMREAD_GRAYSCALE))).tolist()) + 1
    print(f"classes: {classes}")

    # DataFrame
    if num_data == None:
        num_data = sum(os.path.isfile(os.path.join(dir_input+'/original', name)) for name in os.listdir(dir_input+'/original')) 
    d = {"imgpath":[dir_input+f"/original/{i}.png" for i in range(num_data)], "labelpath": [dir_input+f"/label/{i}.png" for i in range(num_data)]}
    data = pd.DataFrame(data=d)
    ## shuffle data
    shuffled_data = data.sample(frac=1, random_state=42)  # random_state はシャッフルの再現性を確保するためのもの
    ## train:validation:test = 1-(val_percent)-(test_percent) : val_percent : test_percent
    train_data, temp_data = train_test_split(shuffled_data, test_size=val_percent+test_percent, random_state=seed)
    validation_data, test_data = train_test_split(temp_data, test_size=val_percent/(val_percent+test_percent), random_state=seed)
    ## reset index
    train_df = train_data.reset_index(drop=True)
    val_df = validation_data.reset_index(drop=True)
    test_df = test_data.reset_index(drop=True)
    print("Num of data:")
    print(f"| - training:   {len(train_df)}")
    print(f"| - validation: {len(val_df)}")
    print(f"| - test:       {len(test_df)}")
    # DataLoader
    train_dataset = Dataset(train_df, classes=classes)
    val_dataset = Dataset(val_df, classes=classes)
    test_dataset = Dataset(test_df, classes=classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)


    # GPU, Optimizer
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = UNet_2D(classes=classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)



    # finetune の場合モデルをロード
    if method == "finetune":
        if loaded_model_index == None: # 指定なし。IOU最高のモデルをload
            print("Find best IOU model...")
            num_pretrain_models = len(os.listdir("./models/pretrain"))
            best_index = 0
            best_score = 0.0
            current_score = 0.0
            for i in range(num_pretrain_models):
                model.load_state_dict(torch.load(f"./models/pretrain/pretrain_{i+1}.pth"))
                with torch.no_grad():
                    for i, data in enumerate(train_loader):
                        inputs, labels = data["img"].to(device), data["label"].to(device)   # いずれも [データ数, クラス数, 縦, 横]
                        outputs = model(inputs)                                              # [データ数, クラス数, 縦, 横]

                        ## IOU
                        sigmoid = nn.Sigmoid()
                        outputs = sigmoid(outputs)
                        tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels.to(torch.int), mode='multilabel', threshold=0.5)
                        batch_iou      = smp.metrics.iou_score(tp, fp, fn, tn, reduction="weighted", class_weights=[0.8, 1.0, 1.2])  # micro
                        current_score += batch_iou * batch_size
                    if best_score < current_score:
                        best_score = current_score
                        best_index = i+1
            model.load_state_dict(torch.load(f"./models/pretrain/pretrain_{best_index}.pth"))
            print(f"Loaded pretrained model: ./models/pretrain/pretrain_{best_index}.pth")
            print(f"                    IOU: {best_score}")
        else: # 指定あり。
            model.load_state_dict(torch.load(f"./models/pretrain/pretrain_{loaded_model_index}.pth"))
            print(f"Loaded pretrained model: ./models/pretrain/pretrain_{loaded_model_index}.pth")
    


    # Loss function
    if loss_type    == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif loss_type  == "JaccardLoss":
        criterion = smp.losses.JaccardLoss(mode='multilabel')
    elif loss_type  == "DiceLoss":
        criterion = smp.losses.DiceLoss(mode='multilabel')
            
    print(f"Loss funtion: {loss_type}")

    # Training
    epoch_count = 0
    pre_iou = -1.0
    ## グラフ + 最適モデル判定 用
    history = {"train_loss": [], "val_loss": [], "batch_iou_micro": [], "batch_iou_class": [], "average_iou_micro": [], "average_iou_class": []}
    for epoch in range(epochs):
        model.train()

        epoch_iou_micro = 0
        epoch_iou_class = torch.zeros(classes).to(device)

        print("-----------------------------------------")
        print(f"epoch: {epoch+1}")
        print("[training]")
        for i, data in enumerate(train_loader):
            inputs, labels = data["img"].to(device), data["label"].to(device)  # [データ数, クラス数, 縦, 横]
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            print(f"| - batch: {i+1}  loss: {loss.item():.5f}")
            history["train_loss"].append(loss.item())

        model.eval()
        sigmoid = nn.Sigmoid()
        print("\n[validation]")
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data["img"].to(device), data["label"].to(device)   # いずれも [データ数, クラス数, 縦, 横]
                outputs = model(inputs)                                              # [データ数, クラス数, 縦, 横]

                # 性能評価
                ## Loss
                loss = criterion(outputs, labels)  
                history["val_loss"].append(loss.item())
                ## Metrics
                outputs = sigmoid(outputs)
                tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels.to(torch.int), mode='multilabel', threshold=0.5)
                ### accuracy
                batch_accuracy  = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
                ### f1
                batch_f1        = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                ### iou
                batch_iou_micro = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                batch_iou_class = smp.metrics.iou_score(tp.sum(0), fp.sum(0), fn.sum(0), tn.sum(0), reduction="none") # このバッチのclass別IOU [class0, class1, class2, ...]
                
                ## 記録
                history["batch_iou_micro"].append(batch_iou_micro.item())
                history["batch_iou_class"].append(batch_iou_class.tolist())   #バッチごとのクラス別IOU 

                ## エポックの合計
                epoch_iou_micro += batch_iou_micro.item() * len(inputs)
                epoch_iou_class += batch_iou_class * len(inputs)
                
                ## 標準出力
                print(f"| - batch: {i+1}\n\t| - loss:      {loss.item():.5f}")
                print(f"\t| - Accuracy:  {batch_accuracy.item():.5f}")
                print(f"\t| - F1 score:  {batch_f1.item():.5f}")
                print(f"\t| - IOU")
                print(f"\t\t| - micro: {batch_iou_micro.item():.5f}")
                print(f"\t\t| - class:")
                for c in range(classes):
                    print(f"\t\t\t| - class {c}: {batch_iou_class[c]:.5f}")
        
        # Average in epoch
        ## 計算
        average_iou_micro = epoch_iou_micro / len(val_df)
        average_iou_class = epoch_iou_class / len(val_df)
        ## 記録
        history["average_iou_micro"].append(average_iou_micro)
        history["average_iou_class"].append([average_iou_class[c] for c in range(classes)])
        ## 標準出力
        print("| - Average:")
        print("\t| - IOU")
        print(f"\t\t| - micro: {average_iou_micro:.5f}")    
        print(f"\t\t| - class")    
        for c in range(classes):
            print(f"\t\t\t| - class {c}: {average_iou_class[c]:.5f}")
        print()

        torch.save(model.state_dict(), f"./models/{method}/{method}_{epoch+1}.pth")

        # Early Stopping
        if average_iou_micro < pre_iou:  # 悪化した場合
            epoch_count += 1
            if epoch_count >= patience and early_stopping:
                print("Early stopped.")
                print("-----------------------------------------")
                break
        else:  # 精度が改善した場合
            epoch_count = 0
            pre_iou = average_iou_micro





    # Loss, iou
    plt.figure()
    plt.plot(history["train_loss"])
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig(dir_output+"/train_loss.png")

    plt.figure()
    plt.plot(history["val_loss"])
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig(dir_output+"/val_loss.png")

    plt.figure()
    plt.plot(history["batch_iou_micro"])
    plt.xlabel('Batch')
    plt.ylabel('IoU')
    plt.ylim(0, 1.0)
    plt.savefig(dir_output+"/batch_iou_micro.png")

    """
    for c in range(classes):
        plt.figure()
        plt.plot([row[c] for row in history["batch_iou_class"]])
        plt.xlabel('batch')
        plt.ylabel('iou of class' + str(c))
        plt.ylim(0, 1.0)
        plt.savefig(dir_output+'/batch_iou_class'+str(c)+'.png')
    """
    plt.figure()
    plt.plot([row[0] for row in history["batch_iou_class"]], label='Background', color=(0, 0, 0))
    plt.plot([row[1] for row in history["batch_iou_class"]], label='Line', color=(0.4, 0.4, 0.4))
    plt.plot([row[2] for row in history["batch_iou_class"]], label='Triangle', color=(0.8, 0.8, 0.8))
    plt.xlabel('Batch')
    plt.ylabel('Class IoU')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig(dir_output+'/batch_iou_class.png')



    # test
    print("[test]")
    #best_model_index = history["val_iou"].index(max(history["val_iou"])) // math.ceil(len(val_df) / batch_size)
    best_model_index =  history["average_iou_micro"].index(max(history["average_iou_micro"]))           # microのiouのaverage最大
    print(f"Best IOU model: {best_model_index+1}")
    print(f"| - IOU\n\t| - micro: {history['average_iou_micro'][best_model_index]:.5f}")
    print("\t| - class")
    for c in range(classes):
        print(f"\t\t| - class {c}: {history['average_iou_class'][best_model_index][c]:.5f}")        
    model.load_state_dict(torch.load(f"./models/{method}/{method}_{best_model_index+1}.pth"))

    model.eval()
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data["img"].to(device), data["label"].to(device)
            outputs = model(inputs)
            #loss = criterion(outputs, labels)
            #print("loss: ",loss.item())

            outputs = sigmoid(outputs)
            pred = torch.argmax(outputs, dim=1)
            pred = torch.nn.functional.one_hot(pred.long(), num_classes=classes).to(torch.float32)

            orig_np = inputs[0,0,:,:].cpu().numpy()
            cv2.imwrite(dir_output+f"/result/{i}_original.png", orig_np*255)

            #lab_np = labels[0,1,:,:].cpu().numpy()
            lab_np = torch.argmax(labels[0,:,:,:], dim=0).cpu().numpy()
            cv2.imwrite(dir_output+f"/result/{i}_label.png", lab_np*255//(classes-1))

            pred_np = torch.argmax(pred[0,:,:,:], dim=2).cpu().numpy()
            cv2.imwrite(dir_output+f"/result/{i}_pred.png", pred_np*255//(classes-1))
            for j in range(classes):
                if j != 0:
                    pred_np = pred[0,:,:,j].cpu().numpy()
                    cv2.imwrite(dir_output+f"/result/{i}_class{j}.png", pred_np*255)

            integrate_edges(dir_output+f"/result/{i}_class1.png", dir_output+f"/result/{i}_class2.png", filepath_output=dir_output+f"/result/{i}_edge.png")  # 追加
            
    print("Finished test.")
    print("-----------------------------------------")

    return best_model_index+1


if __name__ == "__main__":
    arguments = sys.argv

    if arguments[1] == "pretrain":
    
        method = "pretrain"
        dir_input = "./inputs/simu"
        dir_output = "./outputs/pretrain"

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
        model = UNet_2D(classes=3).to(device)

        train(
            method=method,
            dir_input=dir_input,
            dir_output=dir_output,
            classes=3,
            device=device, 
            model=model,
            num_data=500,
            val_percent=0.1,
            test_percent=0.1,
            loss_type="CrossEntropyLoss",
            epochs=80,
            batch_size=32,
            learning_rate=0.001,
            patience=5,
        )
    
    elif arguments[1] == "finetune":
        method = "finetune"
        dir_input = "./outputs/augment"
        dir_output = "./outputs/finetune"


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = UNet_2D(classes=3).to(device)

        train(
            method=method,
            dir_input=dir_input,
            dir_output=dir_output,
            classes=3,
            device=device, 
            model=model,
            loaded_model_index=29, # 経験的にこれは必要
            val_percent=0.1,
            test_percent=0.1,
            loss_type="DiceLoss",
            epochs=80,
            batch_size=4,
            learning_rate=0.0001,
            patience=5,
        )
    