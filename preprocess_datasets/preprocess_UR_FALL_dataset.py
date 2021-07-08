import numpy as np
import cv2
import os
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import crop_movement
from mhi import create_MHI

SRC_DIR = '/home/arkady_big/Repositories/HumanFallDetection/UR_FALL_dataset/dataset'
DEST_DIR = '/home/arkady_big/Repositories/HumanFallDetection/UR_FALL_dataset/dataset_movement_selected'

def select_movements(video_path):
    image_names = os.listdir(video_path)
    
    if len(image_names) < 80:
        return None
    image_names.sort()
    image_names = [osp.join(video_path, i) for i in image_names]
    a = cv2.imread(image_names[80])
    plt.imshow(a)
    plt.show()
    list_mhi = create_MHI(image_names, interval=2, dim=None, use_body_segmentation=False)
    list_mhi = [i[1] for i in list_mhi]
    plt.imshow(list_mhi[0])
    plt.show()
    
    
    movements = [crop_movement(i) for i in list_mhi]
    plt.imshow(movements[0])
    plt.show()
    return movements

classes = os.listdir(SRC_DIR) # /train /test /val
for _class in classes:
    path = osp.join(SRC_DIR, _class)
    videos = os.listdir(path) # /fall /not_fall
    for video_name in videos:
        video_path = osp.join(path, video_name)
        dest_video = osp.join(DEST_DIR, _class, video_name)
        os.makedirs(dest_video, exist_ok=True)
        movements = select_movements(video_path)
        if movements is not None:
            for i, img in enumerate(movements):
                dest_img = osp.join(dest_video, f'{i:0>5}.jpg')
                cv2.imwrite(dest_img, img)
