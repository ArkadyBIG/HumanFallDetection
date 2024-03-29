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

split_batch_size = 1
class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    # def __init__(self, *a, **k):
    #     super(SimpleConfig, self).__init__()
    NAME = "coco_inference"

    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = split_batch_size
    BATCH_SIZE = split_batch_size
    # Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)


# %%
model = None
def init_model():
    global model
    model = mrcnn.model.MaskRCNN(mode="inference",
                                config=SimpleConfig(),
                                model_dir=os.getcwd())

    # Load the weights into the model.
    # Download the mask_rcnn_coco.h5 file from this link: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
    model.load_weights(filepath="/home/kotik/HumanFallDetection/models/mask_rcnn_coco.h5",
                    by_name=True)


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


init_model()

def _inferance(X):
    # split_batch_size = BATCH_SIZE
    X = [X[i * split_batch_size: (i + 1) * split_batch_size] for i in range(len(X) // split_batch_size + 1)]
    X = [x for x in X if x]

    # Last batch still split_batch_size
    addons = split_batch_size - len(X[-1])
    X[-1].extend([np.zeros_like(X[-1][0]) for _ in range(addons)])
    
    Y = []
    for x in tqdm(X):
        y = model.detect(x, verbose=0)
        Y.append(y)
    Y = np.concatenate(Y, axis=0)
    return Y[:-addons or None]
def get_body_masks(imgs_rgb):
    result = _inferance(imgs_rgb)
    masks = []
    for i in range(len(imgs_rgb)):
        r = result[i]
        r['masks'] = r['masks'].transpose(2, 0, 1)
        r = filter_people(r)
        if not r['masks'].shape[0]:
            mask = np.zeros_like(imgs_rgb[i][..., 0], 'bool')
        else:
            box, _, _, mask = get_biggets_result_by_area(r)
        masks.append(mask)
    return masks

def get_body_mask(img_rgb):
    r = model.detect([img_rgb], verbose=0)[0]
    r['masks'] = r['masks'].transpose(2, 0, 1)
    r = filter_people(r)

    if not r['masks'].shape[0]:
        return None

    box, _id, score, mask = get_biggets_result_by_area(r)
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
