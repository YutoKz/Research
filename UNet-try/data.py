import pandas as pd
import numpy as np
import PIL as Image
import cv2

"""

/home/yuto_kizawa/Project/Research/UNet-try/train/case123/case123_day20/scans/
slice_0065_266_266_1.50_1.50.png

case123_day20_slice_0065    ←この3つのindexで一意に指定可能

"""


classes = {"background" : 0, "large_bowel" : 1, "small_bowel" : 2, "stomach" : 3}

df = pd.read_csv("/home/yuto_kizawa/Project/Research/UNet-try/with_perfect_label.csv")
ids = df.id.unique().tolist()
imgpath_list = pd.read_csv("/home/yuto_kizawa/Project/Research/UNet-try/imgpath_hw.csv")["imgpath"].tolist()

labelpath_list = []

for i, id in enumerate(ids):
    df_now = df[df.id == id].reset_index(drop=True)

    imgpath = imgpath_list[i]
    #img = Image.open(imgpath)
    #img = np.asarray(img)
    img = cv2.imread(imgpath)
    img = np.array(img)
    shape = img.shape
    height = shape[0]
    width = shape[1]

    mask = np.zeros(height * width, dtype=np.uint8) # 目的の画像

    for j in range(len(df_now)):
        c = df_now.loc[j, "class"]
        s = df_now.loc[j, "segmentation"]
        s = s.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            mask[lo:hi] =  classes[c]

    mask = mask.reshape((height, width)).clip(0, 3)
    #mask_png = Image.fromarray(mask.astype('uint8'))
    labelpath = "/home/yuto_kizawa/Project/Research/UNet-try/label/" + id + ".png"
    #mask_png.save(labelpath)
    cv2.imwrite(labelpath, mask)
    labelpath_list.append(labelpath)

df_labelpath = pd.DataFrame(labelpath_list)
df_labelpath.to_csv("labelpath.csv", index=False)
