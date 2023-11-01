import h5py

LABELS: list[str] = ["start", "end", "kill", "death", "other"]
DATASET_PATH = "s2_dataset.hdf5"
BATCH_SIZE = 50
TRAIN_SIZE = 47350
TEST_SIZE = 5000
TRAIN_BATCH_COUNT = TRAIN_SIZE // BATCH_SIZE
TEST_BATCH_COUNT = TEST_SIZE // BATCH_SIZE


class S2Data:
    def __init__(self):
        h = h5py.File(DATASET_PATH, "r")
        self.train_xs = h["train_xs"]
        self.train_ys = h["train_ys"]
        self.test_xs = h["test_xs"]
        self.test_ys = h["test_ys"]

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
            if batch_index >= TEST_BATCH_COUNT:
                batch_index = 0
