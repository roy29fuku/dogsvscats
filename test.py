'''
画像を判定する(1 = dog, 0 = cat)
$ python test.py IMAGES
'''

import ingre_cnn as ingre
import sys, os
from PIL import Image
import numpy as np

# コマンドラインからファイル名を得る、引数がなければ終了
if len(sys.argv) <= 1:
    print("ingre-checker.py (ファイル名)")
    quit()

image_size = 50
categories = [
    'dog',
    'cat',
]

# 入力画像をNumpyに変換 --- (※2)
X = []
files = []
for fname in sys.argv[1:]:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    in_data = np.asarray(img)
    X.append(in_data)
    files.append(fname)
X = np.array(X)

# CNNのモデルを構築 --- (※3)
model = ingre.build_model(X.shape[1:])
model.load_weights("./models/dogcat-cnn-model.hdf5")

# データを予測 --- (※4)
pre = model.predict(X)
for i, p in enumerate(pre):
    y = p.argmax()
    print("input:", files[i])
    print("sp:", categories[y])
