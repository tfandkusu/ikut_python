# %%
import os
import cv2
import tensorflow as tf
import numpy as np

# %%
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# %%
labels = ["start", "other", "end", "death", "kill"]
# %%
path = os.path.join("data/01_src_movie/src2.mp4")
# 動画を読み込む
cap = cv2.VideoCapture(path)
# 1秒あたりフレーム数を取得
fps = cap.get(cv2.CAP_PROP_FPS)
# %%
# 0.5秒に1枚の割合でフレームを保存する
skip = fps // 2
# フレームインデックス
i = 0
# 保存インデックス
save_index = 2000
# 保存先ディレクトリ
dst_dir = "data/02_img"
# %%
while True:
    ret, img = cap.read()
    if ret:
        if i % skip == 0:
            # フレームを縮小して保存する
            shrink = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            # 4次元に変換する
            input_tensor = shrink.reshape(1, 224, 224, 3)
            # それをTensorFlow liteに指定する
            interpreter.set_tensor(input_details[0]["index"], input_tensor)
            # 推論実行
            interpreter.invoke()
            # 出力層を確認
            output_tensor = interpreter.get_tensor(output_details[0]["index"])
            # シーン判定
            scene = np.argmax(output_tensor)
            label = labels[scene]
            # 保存する
            dst_path = os.path.join(dst_dir, "%05d_%s.jpg" % (save_index, label))
            cv2.imwrite(dst_path, img)
            save_index += 1
        i += 1
    else:
        break

# %%
