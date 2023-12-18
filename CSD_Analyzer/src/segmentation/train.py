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
from torcheval.metrics.functional import multiclass_accuracy
from sklearn.model_selection import train_test_split
import os
import shutil
import sys
import math
from typing import Literal
from dataset import Dataset
from model import UNet_2D

MethodType = Literal["pretrain", "finetune"]
LossType = Literal["CrossEntropyLoss", "JaccardLoss", "DiceLoss"]


def train(
    method: MethodType,
    dir_input,
    dir_output,
    classes,
    device, 
    model: nn.Module,
    num_data: int = None,
    val_percent: float = 0.1,
    test_percent: float = 0.1,
    loss_type: LossType = "CrossEntropyLoss",
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-3,
): 
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
        num_data:       学習用データの一部を使う場合に使用
        val_percent:    学習用データ内のvalidation dataの割合
        test_percent:   学習用データ内のtest dataの割合
        epochs:
        batch_size: 
        learning_rate: 

    """
    print(f"--- mode: {method} ---")

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
    print(f"num of classes: {classes}")

    # DataFrame
    if num_data == None:
        num_data = int(sum(os.path.isfile(os.path.join(dir_input+'/original', name)) for name in os.listdir(dir_input+'/original')) / 2) 
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
    print(f"num of training data:   {len(train_df)}")
    print(f"num of validation data: {len(val_df)}")
    print(f"num of test data:       {len(test_df)}")
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
    """
    # IOUが最大のモデルを使うため、ここで最適なモデルを判定
    if method == "finetune":
        num_pretrain_models = len(os.listdir("./models/pretrain"))
        best_index = 0
        best_score = 0.0
        current_score = 0.0
        for i in range(num_pretrain_models):
            model.load_state_dict(torch.load(f"./models/pretrain/pretrain_{i+1}.pth"))
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    inputs, labels = data["img"].to(device), data["label"].to(device)   # いずれも [データ数, クラス数, 縦, 横]
                    outputs = model(inputs)                                              # [データ数, クラス数, 縦, 横]

                    ## IOU
                    sigmoid = nn.Sigmoid()
                    outputs = sigmoid(outputs)
                    tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels.to(torch.int), mode='multilabel', threshold=0.5)
                    batch_iou      = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                    current_score += batch_iou * batch_size
                if best_score < current_score:
                    best_score = current_score
                    best_index = i+1
        print(f"Model loaded from ./models/pretrain/pretrain_{best_index}.pth")
        print(f"IOU: {best_score}")
        model.load_state_dict(torch.load(f"./models/pretrain/pretrain_{best_index}.pth"))
        
    """
    if method == "finetune":
        model.load_state_dict(torch.load(f"./models/pretrain/pretrain_29.pth"))
        print(f"Model loaded from ./models/pretrain/pretrain_29.pth")
    


    # Loss function
    if loss_type == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "JaccardLoss":
        criterion = smp.losses.JaccardLoss(mode='multilabel')
    elif loss_type == "DiceLoss":
        criterion = smp.losses.DiceLoss(mode='multilabel')
            
    print(f"Loss funtion: {loss_type}")

    # Training
    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    for epoch in range(epochs):
        model.train()

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
        print("[validation]")
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data["img"].to(device), data["label"].to(device)   # いずれも [データ数, クラス数, 縦, 横]
                outputs = model(inputs)                                              # [データ数, クラス数, 縦, 横]

                # 性能評価
                ## Loss
                loss = criterion(outputs, labels)  
                history["val_loss"].append(loss.item())
                ## Accuracy, IOU
                outputs = sigmoid(outputs)
                tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels.to(torch.int), mode='multilabel', threshold=0.5)
                batch_accuracy  = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
                batch_f1        = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                batch_iou       = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                history["val_iou"].append(batch_iou.item())

                print(f"| - batch: {i+1}\n\t| - loss:      {loss.item():.5f}")
                print(f"\t| - Accuracy:  {batch_accuracy.item():.5f}")
                print(f"\t| - F1 score:  {batch_f1.item():.5f}")
                print(f"\t| - IOU score: {batch_iou.item():.5f}")

        print()
        torch.save(model.state_dict(), f"./models/{method}/{method}_{epoch+1}.pth")
    print("finish training")



    # Loss
    plt.figure()
    plt.plot(history["train_loss"])
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.savefig(dir_output+"/train_loss.png")
    plt.figure()
    plt.plot(history["val_loss"])
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.savefig(dir_output+"/val_loss.png")
    plt.figure()
    plt.plot(history["val_iou"])
    plt.xlabel('batch')
    plt.ylabel('iou')
    plt.savefig(dir_output+"/val_iou.png")






    # test
    print("[test]")
    # model = UNet_2D(classes=classes)
    best_model_index = history["val_iou"].index(max(history["val_iou"])) // math.ceil(len(val_df) / batch_size)
    #model.load_state_dict(torch.load(f"./models/{method}/{method}_{epochs}.pth"))
    model.load_state_dict(torch.load(f"./models/{method}/{method}_{best_model_index+1}.pth"))
    print(f"load model {best_model_index+1}")

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
    print("finish test")


if __name__ == "__main__":
    arguments = sys.argv

    if arguments[1] == "pretrain":
    
        method = "pretrain"
        dir_input = "./data/input_simu"
        dir_output = "./data/output_pretrain"

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = UNet_2D(classes=3).to(device)

        train(
            method=method,
            dir_input=dir_input,
            dir_output=dir_output,
            classes=3,
            device=device, 
            model=model,
            num_data=500,
            val_percent=0.2,
            test_percent=0.2,
            loss_type="CrossEntropyLoss",
            epochs=30,
            batch_size=32,
            learning_rate=0.001,
        )
    
    elif arguments[1] == "finetune":
        method = "finetune"
        dir_input = "./data/input_hitachi/dataset1217"
        dir_output = "./data/output_finetune"


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = UNet_2D(classes=3).to(device)

        train(
            method=method,
            dir_input=dir_input,
            dir_output=dir_output,
            classes=3,
            device=device, 
            model=model,
            val_percent=0.2,
            test_percent=0.2,
            loss_type="DiceLoss",
            epochs=30,
            batch_size=2,
            learning_rate=0.00001,
        )
    