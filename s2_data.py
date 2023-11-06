import h5py

LABELS = ["start", "end", "kill", "death", "other"]
INPUT_IMAGE_SIZE = 224
INPUT_IMAGE_CHANNEL = 3
DATASET_PATH = "s2_dataset.hdf5"
DATASET_TRAIN_XS = "train_xs"
DATASET_TRAIN_YS = "train_ys"
DATASET_TEST_XS = "test_xs"
DATASET_TEST_YS = "test_ys"
MODEL_DIR = "data/04_s2_savedmodel/"
BATCH_SIZE = 50
TRAIN_SIZE = 47350
TEST_SIZE = 5000
TRAIN_BATCH_COUNT = TRAIN_SIZE // BATCH_SIZE
TEST_BATCH_COUNT = TEST_SIZE // BATCH_SIZE


class S2Data:
    def __init__(self):
        # HDF5ファイルを開き、データセットを読み込む
        h = h5py.File(DATASET_PATH, "r")
        self.train_xs = h[DATASET_TRAIN_XS]
        self.train_ys = h[DATASET_TRAIN_YS]
        self.test_xs = h[DATASET_TEST_XS]
        self.test_ys = h[DATASET_TEST_YS]

    def generator(self):
        "訓練データのジェネレータ"
        batch_index = 0
        while True:
            xs = self.train_xs[  # type: ignore
                batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE
            ]
            ys = self.train_ys[  # type: ignore
                batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE
            ]
            yield (xs, ys)
            batch_index += 1
            # 訓練データをすべて使い切ったら、最初からやり直す
            if batch_index >= TRAIN_BATCH_COUNT:
                batch_index = 0

    def generator_validation_data(self):
        "テストデータのジェネレータ"
        batch_index = 0
        while True:
            xs = self.test_xs[  # type: ignore
                batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE
            ]
            ys = self.test_ys[  # type: ignore
                batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE
            ]
            yield (xs, ys)
            batch_index += 1
            # テストデータをすべて使い切ったら、最初からやり直す
            if batch_index >= TEST_BATCH_COUNT:
                batch_index = 0
