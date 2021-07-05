#%%
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5024)])
    except Exception as e: print(e)
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mhi import create_MHI
import torch
from fd_net import load_fd_net_model, inference
from PIL import Image
from human_mask import get_body_mask

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_fd_net_model().to(device)
model.eval()
# %%+
path = './raw_frames/Jenia/'
image_paths = [path + n for n in os.listdir(path)]
image_paths.sort()

images = [cv2.imread(n) for n in image_paths]
masks = get_body_mask(images)

input_batch = create_MHI(image_paths, masks, interval=2)

#%%
preprocessed = [i[1] for i in input_batch]
for i, img in enumerate(preprocessed):
    cv2.imwrite('segmentation/no_segmentation/' + f'{i}.jpg', img)
plt.imshow(input_batch[-1][0])
#%%
# input_batch = [(f, cv2.resize(np.hstack([ np.zeros_like(p) , p]), (224, 224))) for f, p in input_batch]
preprocessed = np.array(preprocessed, 'f4')
preprocessed = preprocessed / 255
preprocessed = preprocessed - np.array([0.485, 0.456, 0.406], 'f4')
preprocessed = preprocessed / np.array([0.229, 0.224, 0.225], 'f4')
preprocessed = preprocessed.transpose(0, 3, 1, 2)
preprocessed.shape
#%%

result = inference(model, preprocessed, split_batch_size=32)
# %%
for (frame, mhi), y in zip(input_batch, result):
    if y[0] > 0.10:
        img = np.hstack([cv2.resize(frame, (224, 224)), mhi])
        plt.imshow(img)
        plt.show()
        print(f'fall: {y[0]:.1%}')

# %%
result - result1


# %%
