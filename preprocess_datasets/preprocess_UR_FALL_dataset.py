#%%
import numpy as np
import cv2
import os
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import crop_movement
from mhi import create_MHI
import random
import shutil
from tqdm import tqdm


SRC_DIR = '/home/kotik/HumanFallDetection/UR_FALL_dataset/dataset'
DEST_DIR = '/home/kotik/HumanFallDetection/UR_FALL_dataset/dataset_movement_selected'

def select_movements(video_path, use_body_segmentation):
    image_names = os.listdir(video_path)
    
    if len(image_names) < 80:
        return None
    image_names.sort()
    image_names = [osp.join(video_path, i) for i in image_names]
    a = cv2.imread(image_names[80])
    plt.imshow(a)
    plt.show()
    list_mhi = create_MHI(image_names, interval=2, dim=None, use_body_segmentation=use_body_segmentation)
    list_mhi = [i[1] for i in list_mhi]
    plt.imshow(list_mhi[36])
    plt.show()
    
    
    movements = [crop_movement(i) for i in list_mhi]
    plt.imshow(movements[36])
    plt.show()
    return movements
#%%
classes = os.listdir(SRC_DIR) # /train /test /val
for _class in classes:
    path = osp.join(SRC_DIR, _class)
    videos = os.listdir(path) # /fall /not_fall
    for video_name in videos:
        video_path = osp.join(path, video_name)
        dest_video = osp.join(DEST_DIR, _class, video_name)
        use_body_segmentation = ((_class  == 'fall') and ('20' in video_name)) 
        movements = select_movements(video_path, use_body_segmentation)
        if movements is not None:
            os.makedirs(dest_video, exist_ok=True)
            for i, img in enumerate(movements):
                dest_img = osp.join(dest_video, f'{i:0>5}.jpg')
                cv2.imwrite(dest_img, img)

# %%

val_split = 0.2
annotation_map = {
    'VideosFall': 'fall',
    'VideosNoFall': 'not_fall'
}

for _class in ['VideosFall', 'VideosNoFall']:
    path = osp.join(DEST_DIR, _class)
    videos = [osp.join(path, v) for v in os.listdir(path)]
    random.shuffle(videos)
    val_end_split = int(len(videos) * val_split)
    val_videos = videos[:val_end_split]
    train_videos = videos[val_end_split:]
    
    for _type, videos in zip(['train', 'val'], tqdm([train_videos, val_videos])):
        dest = osp.join(DEST_DIR, _type, annotation_map[_class], )
        # os.makedirs(dest, exist_ok=True)
        for video in videos:
            # print(video, osp.join(dest, osp.split(video)[-1]))
            shutil.copytree(video, osp.join(dest, osp.split(video)[-1]))


# %%
