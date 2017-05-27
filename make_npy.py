'''
画像をベクトルに変換してnpyファイルに保存しておく。
毎回画像を読み込まなくて済む。
$ python make_npy.py
'''

import glob
import random
import math
from PIL import Image
import numpy as np

load_dir = './images/train/'
save_dir = './images/created/'
categories = [
    'cat',
    'dog',
]
cat_len = len(categories)
image_size = 50

X = []
Y = []
def add_sample(cat, fname):
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)

def make_sample(files):
    global X, Y
    X = []; Y = []
    for cat, fname in files:
        add_sample(cat, fname)
    return np.array(X), np.array(Y)

# ディレクトリごとに分けられたファイルを収集する
allfiles = []
for idx, cat in enumerate(categories):
    image_dir = load_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((idx, f))

# シャッフルして学習データとテストデータに分ける
random.shuffle(allfiles)
th = math.floor(len(allfiles) * 0.6)
train = allfiles[0:th]
test = allfiles[th:]
X_train, y_train = make_sample(train)
X_test, y_test = make_sample(test)
xy = (X_train, X_test, y_train, y_test)
np.save("./images/dogcat.npy", xy)
print("ok,", len(y_train))
