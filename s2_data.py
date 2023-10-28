import csv
import os
from label_path import LabelPath
import cv2
import random
import tensorflow as tf
import numpy as np

LABELS: list[str] = ["start", "end", "kill", "death", "other"]
CSV_PATH = "data/s24_auto_ml.csv"
IMAGE_DIR = "data/03_s2train/"
BATCH_SIZE = 50


class S2Data:
    def __init__(self):
        all_images: list[LabelPath] = []
        with open(CSV_PATH) as f:
            for row in csv.reader(f):
                uri = row[0]
                label = row[1]
                if label in LABELS:
                    filename = uri.split("/")[-1]
                    path = os.path.join(IMAGE_DIR, filename)
                    all_images.append(LabelPath(label=label, path=path))
        # all_images を訓練データとテストデータに分ける
        self.train = all_images[:-5000]
        self.test = all_images[-5000:]

    def generator(self):
        "訓練データのジェネレータ"
        while True:
            xs, ys = self.make_batch()
            yield (xs, ys)

    def make_batch(self):
        "訓練データのバッチを作成する"
        xs = []
        ys = []
        for i in range(BATCH_SIZE):
            x, y = self.make_xy()
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def make_xy(self):
        "訓練データの入力x, 出力yのペアを作成する"
        # ランダムに訓練データを選ぶ
        index = random.randint(0, len(self.train) - 1)
        # 訓練データの入力x 作成
        img = cv2.imread(self.train[index].path)
        x = cv2.resize(img, (224, 224))
        # 訓練データの入力y 作成
        label = self.train[index].label
        y = tf.keras.utils.to_categorical(LABELS.index(label), len(LABELS))
        return x, y
