import os
import cv2
import tensorflow as tf
import numpy as np
import s2_data

model: tf.keras.Model = tf.keras.saving.load_model(s2_data.MODEL_DIR)  # type: ignore
path = os.path.join("data/01_src_movie/src2.mp4")
# 動画を読み込む
cap = cv2.VideoCapture(path)
# 1秒あたりフレーム数を取得
fps = cap.get(cv2.CAP_PROP_FPS)
# 0.5秒に1枚の割合でフレームを保存する
skip = fps // 2
# フレームインデックス
i = 0
# 保存インデックス
save_index = 0
# 保存先ディレクトリ
dst_dir = "data/02_img"
while True:
    ret, img = cap.read()
    if ret:
        if i % skip == 0:
            # フレームを縮小して保存する
            shrink = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            # 4次元に変換する
            input_tensor = shrink.reshape(1, 224, 224, 3)
            # 推論実行
            output_tensor = model.predict(input_tensor)
            # シーン判定
            label_index = np.argmax(output_tensor, axis=1)[0]
            label = s2_data.LABELS[label_index]
            # 保存する
            dst_path = os.path.join(dst_dir, "%05d_%s.jpg" % (save_index, label))
            cv2.imwrite(dst_path, img)
            save_index += 1
        i += 1
    else:
        break
