
import glob
from pathlib import Path
import imageio
import numpy as np
import os
from PIL import Image
import cv2
import shutil
import pandas as pd

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')

# ======================================================================================================================
# Adapt data from Zenkl et al., 2022
# ======================================================================================================================

workdir = "C:/Users/anjonas/PycharmProjects/EWS_Herbifly"

# existing file locations
train_imgs = glob.glob(f'{workdir}/data/train/*[0-9].png')
train_masks = glob.glob(f'{workdir}/data/train/*_mask.png')
val_imgs = glob.glob(f'{workdir}/data/validation/*[0-9].png')
val_masks = glob.glob(f'{workdir}/data/validation/*_mask.png')
test_imgs = glob.glob(f'{workdir}/data/test/*[0-9].png')
test_masks = glob.glob(f'{workdir}/data/test/*_mask.png')
imgs = train_imgs + val_imgs + test_imgs
msks = train_masks + val_masks + test_masks

# train and test go to the new train, validation goes to validation
keys = ["train"] * len(train_imgs) + ["validation"] * len(val_imgs) + ["train"] * len(test_imgs)
# new file locations
new_data_dir = Path(f'{workdir}/data2')
new_dirs = [Path(f'{new_data_dir}/train/images'), Path(f'{new_data_dir}/train/masks'),
            Path(f'{new_data_dir}/validation/images'), Path(f'{new_data_dir}/validation/masks')]
for dir in new_dirs:
    dir.mkdir(exist_ok=True, parents=True)

new_train_imgs = train_imgs + test_imgs
new_validation = val_imgs

# move to new directory
for i in range(len(keys)):

    print(i)

    # IMAGE
    base_name = os.path.basename(imgs[i])
    image = Image.open(imgs[i])
    image = np.asarray(image)
    imageio.imwrite(f'{new_data_dir}/{keys[i]}/images/{base_name}', image)

    # MASK
    mask = Image.open(msks[i])
    mask = mask.convert('L')
    mask = np.asarray(mask)
    mask = mask / 255.0
    # Encode the greyscale masks
    mask = np.where(mask > 0.5, np.ones_like(mask), np.zeros_like(mask))
    # invert
    mask = np.where(mask == 1, 0, 1).astype("uint8")
    imageio.imwrite(f'{new_data_dir}/{keys[i]}/masks/{base_name}', mask)

# ======================================================================================================================
# Adapt data from Anderegg et al., 2023
# ======================================================================================================================

base_dir = "/home/anjonas/kp-public/Evaluation/Hiwi/2023_herbifly_LTS/raw"
to_dir = "/home/anjonas/kp-public/Evaluation/Projects/KP0011/7/handheld"

images = glob.glob(f'{base_dir}/*/*/Handheld/*/*.JPG')

# 394 unique names
basenames = [os.path.basename(x) for x in images]

for i in images:
    print(i)
    base_name = os.path.basename(i)
    try:
        shutil.copy(i, f'{to_dir}/{base_name}', )
    except PermissionError:
        print("permission denied")
        continue

# ======================================================================================================================
# Adapt data from Madec et al., 2023
# ======================================================================================================================

vegann_path = "/home/anjonas/kp-public/Evaluation/Projects/KP0011/7/VegAnn_dataset/VegAnn_dataset"
info = pd.read_csv(f'{vegann_path}/VegAnn_dataset.csv',
                   sep=";")

keys = info["TVT-split1"].tolist()
keys = [x.replace("Test", "Training") for x in keys]

imgs = glob.glob(f'{vegann_path}/images/*.png')
msks = glob.glob(f'{vegann_path}/annotations/*.png')

# new file locations
new_data_dir = Path(f'./data3')
new_dirs = [Path(f'{new_data_dir}/Training/images'), Path(f'{new_data_dir}/Training/masks'),
            Path(f'{new_data_dir}/Validation/images'), Path(f'{new_data_dir}/Validation/masks')]
for dir in new_dirs:
    dir.mkdir(exist_ok=True, parents=True)

for i in range(len(keys)):

    print(i)

    # IMAGE
    base_name = os.path.basename(imgs[i])
    image = Image.open(imgs[i])
    image = np.asarray(image)
    imageio.imwrite(f'{new_data_dir}/{keys[i]}/images/{base_name}', image)

    # MASK
    mask = Image.open(msks[i])
    mask = mask.convert('L')
    mask = np.asarray(mask)
    mask = mask / 255.0
    mask = mask.astype("uint8")
    imageio.imwrite(f'{new_data_dir}/{keys[i]}/masks/{base_name}', mask)

