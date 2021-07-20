
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow_6gb_limit

from time import time

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
import torch
from PIL import Image

from fd_net import inference, load_fd_net_model
from human_mask import get_body_mask, model
from mhi import create_MHI

# %%

# %%
path = './raw_frames/Nastia_fall/'
image_paths = [path + n for n in os.listdir(path)]
image_paths.sort()
len(image_paths)
#%%
# del preprocessed 
def preprocess(frame):
    # frame = cv2.imread(frame)
    # frame = cv2.resize(frame, (300, 300))#, fx=0, fy=0.5)
    # frame = np.hstack([np.zeros_like(frame)[:, :100], frame, np.zeros_like(frame)[:, :100], ])
    # frame = np.vstack([np.zeros_like(frame)[:100], frame, np.zeros_like(frame)[:100], ])
    # center = np.array(frame.shape) / 2
    # h, w = 224, 224
    # x = int(center[1] - w/2)
    # y = int(center[0] - h/2)
    # frame = frame[y:y+h, x:x+w]
    return frame

_input_batch = create_MHI(image_paths, preprocess=preprocess, interval=2, dim=None, use_body_segmentation=1)


#%%
input_batch = [(a, b.copy()) for a, b in _input_batch]
preprocessed = [i[1] for i in input_batch]

from preprocess_datasets.utils import crop_movement

# img = crop_movment(input_batch[n][1])
preprocessed_h = [crop_movement(p) for p in preprocessed]
input_batch = [a for a, b in zip(input_batch, preprocessed_h) if b is not None]
preprocessed_h = [b for b in preprocessed_h if b is not None]
# for i, img in enumerate(preprocessed):
#     cv2.imwrite('segmentation/no_segmentation/' + f'{i}.jpg', img)
#%%
n = 30
plt.imshow(input_batch[n][0])
plt.show()
plt.imshow(input_batch[n][1])
plt.show()
plt.imshow(preprocessed_h[n])
#%%
# input_batch = [(f, cv2.resize(np.hstack([ np.zeros_like(p) , p]), (224, 224))) for f, p in input_batch]
preprocessed = np.array(preprocessed_h, 'f4')
preprocessed = preprocessed / 255
preprocessed = preprocessed - np.array([0.485, 0.456, 0.406], 'f4')
preprocessed = preprocessed / np.array([0.229, 0.224, 0.225], 'f4')
preprocessed = preprocessed.transpose(0, 3, 1, 2)
preprocessed.shape
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = 'train_model/fd_net_augmentation/e_19_val_loss_0.086_acc_0.968.pt'
PATH = 'train_model/movement_selection/e_26_val_loss_0.080_acc_0.972.pt'
PATH = 'train_model/3_datasets/e_29_val_loss_0.085_acc_0.971.pt'
model = load_fd_net_model(PATH).to(device)
_ = model.eval()
#%%
result = inference(model, preprocessed, split_batch_size=32)
result = result[:, 0]
result.max()
#%%

# %%
karnel = np.ones(3)
karnel = karnel / karnel.sum()
_result = result.copy()
_result[_result < 0.8] = 0
_result = np.convolve(_result, karnel, 'same')
plt.plot(_result, )
_result[_result >= 0.8] = 1
_result[_result < 0.8] = 0
karnel = np.ones(10)
_result = np.convolve(_result, karnel, 'same')
_result[_result > 0] = 1
_result = _result.astype(int)
plt.plot(_result, )
#%%
out_path = 'results/3_datasets/Nastia_fall_.mp4'
if os.path.isfile(out_path):
    0 / 0
os.makedirs(out_path.rsplit('/', 1)[0], exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(out_path, fourcc, 20.0, (224 * 3, 224))
for i, ((frame, mhi), y, label, x) in enumerate(zip(input_batch, result, _result, preprocessed_h)):
    img = np.hstack([cv2.resize(frame, (224, 224)), cv2.resize(mhi, (224, 224)), x])
    cv2.putText(img, f'fall: {y:.1%}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 2), thickness=2)
    cv2.putText(img, ['not_fall', 'fall'][label], (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 255)[::1 if label else -1], thickness=2)
    if y > 0.99:
        plt.imshow(img)
        plt.show()
        print(mhi.mean())
    out.write(img)
out.release()

# %%


# %%
