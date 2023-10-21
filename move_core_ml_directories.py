# %%
import csv
import shutil
import os

# %%
with open("data/s24_auto_ml.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        uri = row[0]
        label = row[1]
        filename = uri.split("/")[-1]
        src_file_path = os.path.join("data/03_s2train", filename)
        dst_dir = os.path.join("data/03_s2train", label)
        shutil.move(src_file_path, dst_dir)


# %%
