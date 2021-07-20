import os
import numpy as np
import cv2

from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy import ndimage
from collections import deque

class MHIProcessor:
    '''
    Process MHI as inputs of Fall Detector model
    '''
    def __init__(self, dim=128, threshold=0.1, interval=2, duration=40, use_body_segmentation=True):
        # initialize MHI params
        self.index = 0
        self.dim = dim
        self.threshold = threshold
        self.interval = interval
        self.duration = duration
        self.decay = 1 / self.duration
        self.use_body_segmentation = use_body_segmentation
        self.masks = deque(maxlen=duration)
        #initialize frames
    
    def resize(self, img):
        if self.dim is not None:
            return cv2.resize(img,(self.dim, self.dim),
                                         interpolation=cv2.INTER_AREA)
        return img.copy()
    def process(self, frame_bgr):
        self.index += 1

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.index == 1:
            self.prev_frame = self.resize(frame)
            self.prev_mhi = np.zeros_like(self.prev_frame)
            
        if self.index % self.interval == 0:
            frame = self.resize(frame)
            diff = cv2.absdiff(self.prev_frame, frame)
            
            binary = (diff >= (self.threshold * 255)).astype(np.uint8)
            mask = True 
            if self.use_body_segmentation:
                from human_mask import get_body_mask
                mask = get_body_mask(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                if mask is None:
                    binary.fill(0)
                else:
                    mask = self.resize(mask.astype('u1'))
                    binary = mask & binary
            mhi = binary + (binary == 0) * np.maximum(np.zeros_like(self.prev_mhi),
                                                      (self.prev_mhi - self.decay))
            # update frames
            self.prev_frame = frame
            self.prev_mhi = mhi
            
            if self.index >= (self.duration * self.interval) and (mask is not None):
                img = cv2.normalize(mhi, None, 0.0, 255.0, cv2.NORM_MINMAX)
                return cv2.cvtColor(img.astype('u1'), cv2.COLOR_GRAY2BGR)
                
        return None

def create_MHI(images, preprocess=None, use_body_segmentation=True, **k):
    mhi_processor = MHIProcessor(use_body_segmentation=use_body_segmentation, **k)

    preprocessed = []
    for frame in tqdm(images):
        if isinstance(frame, str):
            frame = cv2.imread(frame)
        if preprocess is not None:
            frame = preprocess(frame)
        img = mhi_processor.process(frame)
        # frame_id = mhi_processor.index
 
        if img is not None:
            yield (frame, img)
    return preprocessed


