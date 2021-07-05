import os
import numpy as np
import cv2
# from human_mask import get_body_mask
from tqdm import tqdm
from itertools import zip_longest

import matplotlib.pyplot as plt

class MHIProcessor:
    '''
    Process MHI as inputs of Fall Detector model
    '''
    def __init__(self, dim=128, threshold=0.1, interval=2, duration=40):
        # initialize MHI params
        self.index = 0
        self.dim = dim
        self.threshold = threshold
        self.interval = interval
        self.duration = duration
        self.decay = 1 / self.duration
        
        #initialize frames
        self.mhi_zeros = np.zeros((dim, dim))        
        
    
    def process(self, frame_bgr, mask):
        self.index += 1

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.index == 1:
            self.prev_frame = cv2.resize(frame,(self.dim, self.dim),
                                         interpolation=cv2.INTER_AREA)
            self.prev_mhi = self.mhi_zeros
            
        if self.index % self.interval == 0:
            frame = cv2.resize(frame,(self.dim, self.dim),
                                         interpolation=cv2.INTER_AREA)
            diff = cv2.absdiff(self.prev_frame, frame)
            
            binary = (diff >= (self.threshold * 255)).astype(np.uint8)
            # mask = get_body_mask(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).astype('u1')
            if mask is not None:
                mask = cv2.resize(mask,(self.dim, self.dim))
                binary = mask & binary

            mhi = binary + (binary == 0) * np.maximum(self.mhi_zeros,
                                                      (self.prev_mhi - self.decay))
            # update frames
            self.prev_frame = frame
            self.prev_mhi = mhi
            
            if self.index >= (self.duration * self.interval):
                img = cv2.normalize(mhi, None, 0.0, 255.0, cv2.NORM_MINMAX)
                # plt.imshow(img)
                # plt.show()
                return cv2.cvtColor(img.astype('u1'), cv2.COLOR_GRAY2BGR)
                
        return None

def create_MHI(images, masks=None, **k):
    masks = masks or []
    mhi_processor = MHIProcessor(**k)

    preprocessed = []
    for frame, mask in tqdm(zip_longest(images, masks)):
        if isinstance(frame, str):
            frame = cv2.imread(frame)
        img = mhi_processor.process(frame, mask)
 
        if img is not None:
            img = cv2.resize(img, (224, 224))
            preprocessed.append((frame, img))
    return preprocessed


