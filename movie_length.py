import os
import cv2

# 合計録画秒数
total_seconds = 0
# 元動画格納ディレクトリ
dirname = "/mnt/d/ikut_train/"
# 動画ファイル一覧
for root, dirs, files in os.walk(dirname):
    for name in files:
        if name.endswith(".mkv"):
            path = os.path.join(root, name)
            # 動画を読み込む
            cap = cv2.VideoCapture(path)
            # フレーム数を取得
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # 1秒あたりフレーム数を取得
            fps = cap.get(cv2.CAP_PROP_FPS)
            # 秒数を取得
            seconds = frame_count // fps
            # ファイル名と秒数を出力
            print("%s %d min" % (path, seconds / 60))
            # 合計する
            total_seconds += seconds
# 合計を出力する
print(total_seconds / 3600)
