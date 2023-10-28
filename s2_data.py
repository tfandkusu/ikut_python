import csv
import os
from label_path import LabelPath

LABELS: list[str] = ["start", "end", "kill", "death", "other"]
IMAGE_DIR = "data/03_s2train/"


class S2Data:
    def __init__(self):
        all_images: list[LabelPath] = []
        with open("data/s24_auto_ml.csv") as f:
            for row in csv.reader(f):
                uri = row[0]
                label = row[1]
                if label in LABELS:
                    filename = uri.split("/")[-1]
                    path = os.path.join(IMAGE_DIR, filename)
                    all_images.append(LabelPath(label=label, path=path))
        for image in all_images:
            print(image)
