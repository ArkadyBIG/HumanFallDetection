# %%
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
# %%
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"

    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1
    # Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)


# %%
model = mrcnn.model.MaskRCNN(mode="inference",
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
# Download the mask_rcnn_coco.h5 file from this link: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
model.load_weights(filepath="models/mask_rcnn_coco.h5",
                   by_name=True)

# %%
alpha = 0.5


def filter_people(result):
    result = result.copy()
    ids = np.argwhere(result['class_ids'] == 1).flatten()
    for key in result:
        result[key] = result[key][ids]
    return result


def get_biggets_result_by_area(result):
    el = max(result['masks'], key=np.sum)
    _id = np.where(result['masks'] == el)[0][0]
    return result['rois'][_id], result['class_ids'][_id], result['scores'][_id], result['masks'][_id]


def get_body_mask(img_rgb):
    r = model.detect([img_rgb], verbose=0)[0]
    r['masks'] = r['masks'].transpose(2, 0, 1)
    r = filter_people(r)

    if not r['masks'].shape[0]:
        return np.zeros_like(img_rgb[..., 0], 'bool')

    box, _, _, mask = get_biggets_result_by_area(r)
    return mask
# #%%
# for name in tqdm(os.listdir('raw_frames/Jenia')[6:40]):

#     image = cv2.imread("raw_frames/Jenia/"+name)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Perform a forward pass of the network to obtain the results

#     # Get the results for the first image.
#     r = r[0]


#     result = image
#     count = 0
#     r['masks'] = r['masks'].transpose(2, 0, 1)
#     # for index in range(len(r['class_ids'])):
#     #     if r['class_ids'][index] == 1:
#     #         count+=1
#     #         mask = r['masks'][:, :, index]
#     #         idx = (mask == 1)
#     #         result[idx] = result[idx] * alpha + (1-alpha) * np.array(color)
#     idx = (mask == 1)
#     result[idx] = result[idx] * alpha + (1-alpha) * np.array(color)
#     result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
#     cv2.imwrite('human_masks/'+name, result)
#     break

# %%
