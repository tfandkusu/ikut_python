import csv
import os
import h5py
import numpy as np
import tensorflow as tf
import s2_data
from label_path import LabelPath

CSV_PATH = "data/s24_auto_ml.csv"
IMAGE_DIR = "data/03_s2train/"

all_images: list[LabelPath] = []
with open(CSV_PATH) as f:
    for row in csv.reader(f):
        uri = row[0]
        label = row[1]
        if label in s2_data.LABELS:
            filename = uri.split("/")[-1]
            path = os.path.join(IMAGE_DIR, filename)
            all_images.append(LabelPath(label=label, path=path))
# all_images を訓練データとテストデータに分ける
train = all_images[:-5000]
test = all_images[-5000:]
# HDF5ファイルに書き出す
with h5py.File(s2_data.DATASET_PATH, "w") as h5:
    train_xs = h5.create_dataset(
        "train_xs", shape=(len(train), 224, 224, 3), dtype=np.uint8
    )
    train_ys = h5.create_dataset(
        "train_ys", shape=(len(train), len(s2_data.LABELS)), dtype=np.uint8
    )
    test_xs = h5.create_dataset(
        "test_xs", shape=(len(test), 224, 224, 3), dtype=np.uint8
    )
    test_ys = h5.create_dataset(
        "test_ys", shape=(len(test), len(s2_data.LABELS)), dtype=np.uint8
    )
    for index, label_path in enumerate(train):
        image = tf.keras.utils.load_img(label_path.path, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(image)
        y = tf.keras.utils.to_categorical(
            s2_data.LABELS.index(label_path.label), len(s2_data.LABELS)
        )
        train_xs[index] = x
        train_ys[index] = y
    for index, label_path in enumerate(test):
        image = tf.keras.utils.load_img(label_path.path, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(image)
        y = tf.keras.utils.to_categorical(
            s2_data.LABELS.index(label_path.label), len(s2_data.LABELS)
        )
        test_xs[index] = x
        test_ys[index] = y
