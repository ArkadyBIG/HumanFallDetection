#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mhi import create_MHI
import torch
from fd_net import load_fd_net_model, inference
from PIL import Image
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_fd_net_model().to(device)
model.eval()
# %%

path = './Jenia/'
images = [path + n for n in os.listdir(path)]

input_batch = create_MHI(sorted(images), interval=2)


input_batch = [(f, cv2.resize(np.hstack([ np.zeros_like(p) , p]), (224, 224))) for f, p in input_batch]
preprocessed = [i[1] for i in input_batch]
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
