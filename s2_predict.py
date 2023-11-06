import s2_data
from s2_data import S2Data
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

g = S2Data()
# モデルを SavedModel から読み込む
model: tf.keras.Model = tf.keras.saving.load_model(s2_data.MODEL_DIR)  # type: ignore

# テストデータのジェネレータを取得
generator = g.generator_validation_data()
# すべてのテストデータの正解と予測結果を格納する配列
yss_true = np.zeros((s2_data.TEST_SIZE, len(s2_data.LABELS)), dtype=np.float32)
yss_pred = np.zeros((s2_data.TEST_SIZE, len(s2_data.LABELS)), dtype=np.float32)
# バッチごとに予測する
for batch_index in range(s2_data.TEST_BATCH_COUNT):
    xs, ys_true = next(generator)
    ys_pred = model.predict(xs)
    # 正解を格納する
    yss_true[
        batch_index * s2_data.BATCH_SIZE : (batch_index + 1) * s2_data.BATCH_SIZE
    ] = ys_true
    # 予測結果を格納する
    yss_pred[
        batch_index * s2_data.BATCH_SIZE : (batch_index + 1) * s2_data.BATCH_SIZE
    ] = ys_pred
# 混同行列を作成する
cm = confusion_matrix(yss_true.argmax(axis=1), yss_pred.argmax(axis=1))
# Pandas で混同行列にラベルをつける
df = pd.DataFrame(cm, index=s2_data.LABELS, columns=s2_data.LABELS)
print("Confusion Matrix:")
print(df)
