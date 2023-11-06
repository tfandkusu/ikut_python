import csv
import os
import h5py
import numpy as np
import tensorflow as tf
import s2_data
from label_path import LabelPath
from tqdm import tqdm

CSV_PATH = "data/s24_auto_ml.csv"
IMAGE_DIR = "data/03_s2train/"

# CSV ファイルから画像のパスとラベルを取得する
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
# テストデータは 5000 枚、残りは訓練データ
train = all_images[:-5000]
test = all_images[-5000:]
# HDF5ファイルを作成する
with h5py.File(s2_data.DATASET_PATH, "w") as h5:
    # あらかじめデータセットをサイズを指定して作成する
    # 訓練データの入力
    train_xs = h5.create_dataset(
        s2_data.DATASET_TRAIN_XS,
        shape=(
            len(train),
            s2_data.INPUT_IMAGE_SIZE,
            s2_data.INPUT_IMAGE_SIZE,
            s2_data.INPUT_IMAGE_CHANNEL,
        ),
        dtype=np.uint8,
    )
    # 訓練データの出力
    train_ys = h5.create_dataset(
        s2_data.DATASET_TRAIN_YS,
        shape=(len(train), len(s2_data.LABELS)),
        dtype=np.uint8,
    )
    # テストデータの入力
    test_xs = h5.create_dataset(
        s2_data.DATASET_TEST_XS,
        shape=(
            len(test),
            s2_data.INPUT_IMAGE_SIZE,
            s2_data.INPUT_IMAGE_SIZE,
            s2_data.INPUT_IMAGE_CHANNEL,
        ),
        dtype=np.uint8,
    )
    # テストデータの出力
    test_ys = h5.create_dataset(
        s2_data.DATASET_TEST_YS, shape=(len(test), len(s2_data.LABELS)), dtype=np.uint8
    )
    # データセットに1要素ずつ書き込む
    # 訓練データ
    for index, label_path in enumerate(tqdm(train)):
        # 画像を読み込む
        image = tf.keras.utils.load_img(
            label_path.path,
            target_size=(s2_data.INPUT_IMAGE_SIZE, s2_data.INPUT_IMAGE_SIZE),
        )
        # 入力層の配列に変換する
        x = tf.keras.utils.img_to_array(image, dtype=np.uint8)
        # 出力層の配列に変換する
        # 例: death -> [0, 0, 0, 1, 0]
        y = tf.keras.utils.to_categorical(
            s2_data.LABELS.index(label_path.label), len(s2_data.LABELS)
        )
        # データセットに書き込む
        train_xs[index] = x
        train_ys[index] = y
    # テストデータ
    for index, label_path in enumerate(tqdm(test)):
        image = tf.keras.utils.load_img(
            label_path.path,
            target_size=(s2_data.INPUT_IMAGE_SIZE, s2_data.INPUT_IMAGE_SIZE),
        )
        x = tf.keras.utils.img_to_array(image, dtype=np.uint8)
        y = tf.keras.utils.to_categorical(
            s2_data.LABELS.index(label_path.label), len(s2_data.LABELS)
        )
        test_xs[index] = x
        test_ys[index] = y
