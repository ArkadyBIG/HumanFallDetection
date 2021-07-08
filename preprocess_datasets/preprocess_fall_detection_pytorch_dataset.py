import numpy as np
import cv2
import os
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import crop_movement

ORIGINAL_RESOLUTION = (320, 240)

SRC_DIR = '/home/arkady_big/Repositories/HumanFallDetection/Fall-Detection-PyTorch/dataset'
DEST_DIR = '/home/arkady_big/Repositories/HumanFallDetection/Fall-Detection-PyTorch/dataset_movement_selected'
dataset_types = os.listdir(SRC_DIR) # /train /test /val

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, ORIGINAL_RESOLUTION)
    movement = crop_movement(img)
    return movement

for data_type in dataset_types:
    path = osp.join(SRC_DIR, data_type)
    classes = os.listdir(path) # /fall /not_fall
    for _class in classes:
        videos = osp.join(path, _class)
        dest_videos = osp.join(DEST_DIR, data_type, _class)
        os.makedirs(dest_videos, exist_ok=True)
        for img_name in tqdm(os.listdir(videos)):
            img_path = osp.join(videos, img_name)
            dest_path =  osp.join(dest_videos, img_name)
            movement = preprocess(img_path)
            if movement is not None:
                cv2.imwrite(dest_path, movement)
            break
