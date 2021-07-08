import numpy as np
import cv2

def crop_movement(mhi):
    mhi = np.vstack([np.zeros_like(mhi), mhi, np.zeros_like(mhi)])
    mhi = np.hstack([np.zeros_like(mhi), mhi, np.zeros_like(mhi)])
    thresh = (mhi[..., 0] > 0).astype('u1') * 255
    kernel = np.ones((5, 5))
    thresh = cv2.erode(thresh,kernel,iterations=1)
    kernel = np.ones((10, 10))
    thresh = cv2.dilate(thresh,kernel,iterations=5)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    if not cnts:
        # mhi[..., 1].fill(255)
        # return np.zeros((224, 224, 3), 'u1')
        return None
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
    x = int(max(0, _x))
    y = int(max(0, _y))
    w = int(w)
    h = int(h)
    # print(w / h)
    movement = mhi[y:y+h, x:x+w].copy()
    movement = cv2.resize(movement, (224, 224))
    # cv2.rectangle(mhi, (int(rect[0]), int(rect[1])), \
    # (int(rect[0]+rect[2]), int(rect[1]+rect[3])), (0, 255, 0), 4)
    return movement
