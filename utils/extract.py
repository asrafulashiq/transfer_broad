import numpy as np
import os
import argparse
from tqdm import tqdm
import skimage.io

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=None, type=str)
parser.add_argument("--mode", default="test", type=str)

args = parser.parse_args()

if args.dataset == "miniImagenet":
    data = np.load(f"mini-imagenet/mini-imagenet-cache-{args.mode}.pkl",
                   allow_pickle=True)
    write_root = f"miniImagenet_{args.mode}"
    if not os.path.isdir(write_root):
        os.makedirs(write_root)
    for key in tqdm(data['class_dict']):

        key_dir = os.path.join(write_root, key)
        if not os.path.isdir(key_dir):
            os.makedirs(key_dir)

        indices = data['class_dict'][key]
        for ind in indices:
            image = data['image_data'][ind]

            fname = os.path.join(key_dir, f"{key}_{ind}.png")

            skimage.io.imsave(fname, image)

elif args.dataset == "tieredImagenet":
    pass
