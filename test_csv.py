'''
画像を判定する(1 = dog, 0 = cat)
提出用csvを作成
$ python test.py ./images/test1/*.jpg
'''

import train
import sys, os
from PIL import Image
import numpy as np
import csv


# コマンドラインからファイル名を得る、引数がなければ終了
if len(sys.argv) <= 1:
    print("test.py (ファイル名)")
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
model = train.build_model(X.shape[1:])
model.load_weights("./models/dogcat-cnn-model.hdf5")

# データを予測 --- (※4)
html = ""
pre = model.predict(X)
for i, p in enumerate(pre):
    y = p.argmax()
    print("input:", files[i])
    print("sp:", categories[y])
    html += """
        <h3>input:{0}</h3>
        <div>
          <p><img src="{1}" width=300></p>
          <p>sp:{2}</p>
        </div>
    """.format(os.path.basename(files[i]),
        files[i],
        categories[y])

# レポートを保存 --- (※5)
# html = "<html><body style='text-align:center;'>" + \
#     "<style> p { margin:0; padding:0; } </style>" + \
#     html + "</body></html>"
# with open("result.html", "w") as f:
#     f.write(html)

# CSVに保存
header = ['id', 'label']
data = []
pre = model.predict(X)
for i, p in enumerate(pre):
    y = p.argmax()
    # 反対だった。。。
    if y == 0:
        y = 1
    elif y == 1:
        y = 0
    data.append([i, y])

with open('result.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(data)
