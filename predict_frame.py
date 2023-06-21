# %%
import os
import cv2
import tensorflow as tf

# %%
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.get_input_details()
# %%
path = os.path.join("data/01_src_movie/src.mp4")
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
save_index = 0
# 保存先ディレクトリ
dst_dir = "data/02_img"
while True:
    ret, img = cap.read()
    if ret:
        if i % skip == 0:
            # フレームを縮小して保存する
            shrink = cv2.resize(img, (480, 270), interpolation=cv2.INTER_CUBIC)
            out_path = os.path.join(dst_dir, "frame%05d.jpg" % save_index)
            cv2.imwrite(out_path, shrink)
            print(out_path)
            save_index += 1
        i += 1
    else:
        break

# %%
