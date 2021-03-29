"""
merge train and test of crop disease datasets
"""

import os
from os.path import expanduser

data_path = [
    os.path.expanduser("~/datasets/cdfsl/CropDiseases/test"),
    os.path.expanduser("~/datasets/cdfsl/CropDiseases/train")
]
new_data_path = os.path.expanduser("~/datasets/cdfsl/CropDiseases/all")

for each_path in data_path:
    all_classes = sorted(os.listdir(each_path))
    for each_class in all_classes:
        new_class_path = os.path.join(new_data_path, each_class)
        if not os.path.exists(new_class_path):
            os.makedirs(new_class_path, exist_ok=True)

        for image in os.listdir(os.path.join(each_path, each_class)):
            fullfile = os.path.join(each_path, each_class, image)
            newfile = os.path.join(new_class_path, image)
            os.symlink(fullfile, newfile)
