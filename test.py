
#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6024)])
    except Exception as e: print(e)
from human_mask import get_body_mask, model
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mhi import create_MHI
import torch
from fd_net import load_fd_net_model, inference
from PIL import Image
from time import time
import imutils
import tensorflow.keras as keras
# %%

# %%
path = './raw_frames/Marina_fall/'
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

def crop_movment(mhi):
    mhi = np.vstack([np.zeros_like(mhi), mhi, np.zeros_like(mhi)])
    mhi = np.hstack([np.zeros_like(mhi), mhi, np.zeros_like(mhi)])
    thresh = (mhi[..., 0] > 0).astype('u1') * 255
    kernel = np.ones((10, 10))
    thresh = cv2.dilate(thresh,kernel,iterations = 5)
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        mhi[..., 1].fill(255)
        return np.zeros((224, 224, 3), 'u1')
    c = max(cnts, key=cv2.contourArea)
    cp = cv2.approxPolyDP(c, 3, True)
    rect = cv2.boundingRect(cp)
    img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    _x, _y, w, h = rect
    s = 0.1
    _x -= w * s
    w += w * 2 * s
    _y -= h * s
    h += h * 2 * s
    ratio = w / h
    target = 1
    if ratio < target:
        diff = (h * target - w) / 2
        w += 2 * diff
        _x -= diff
    else:
        diff = (w / target - h) / 2
        _y -= diff
        h += 2 * diff
    # diff = w - h
    # if diff > 0:
    #     y = max(0, y - diff // 2)
    #     h -= diff // 2
    # elif diff < 0:
    #     x = max(0, x + diff // 2)
    #     w += diff // 2
    x = int(max(0, _x))
    y = int(max(0, _y))
    w = int(w)
    h = int(h)
    # print(w / h)
    movement = mhi[y:y+h, x:x+w].copy()
    movement = cv2.resize(movement, (224, 224))
    cv2.rectangle(mhi, (int(rect[0]), int(rect[1])), \
    (int(rect[0]+rect[2]), int(rect[1]+rect[3])), (0, 255, 0), 4)
    return movement
# img = crop_movment(input_batch[n][1])
preprocessed_h = [crop_movment(p) for p in preprocessed]
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
PATH = 'train_model/fdnet_UR_FALL_augmentations_095.pt'
model = load_fd_net_model(PATH).to(device)
_ = model.eval()
#%%
result = inference(model, preprocessed, split_batch_size=32)
result = result[:, 0]
result.max()
#%%
karnel = np.ones(3)
karnel = karnel / karnel.sum()
result = np.convolve(result, karnel, 'same')
# %%
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (224 * 3, 224))
for i, ((frame, mhi), y, x) in enumerate(zip(input_batch, result, preprocessed_h)):
    img = np.hstack([cv2.resize(frame, (224, 224)), cv2.resize(mhi, (224, 224)), x])
    cv2.putText(img, f'fall: {y:.1%}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 2), thickness=2)
    if y > 0.5:
        plt.imshow(img)
        plt.show()
        print(mhi.mean())
    out.write(img)
out.release()
               # %%



# %%
