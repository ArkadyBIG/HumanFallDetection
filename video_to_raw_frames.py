import cv2
import os
path = 'Jenia.mp4'
dest = 'Jenia/'

os.makedirs(dest, exist_ok=True)
cap = cv2.VideoCapture(path)

for i in range(1_000_000):
    ret, frame = cap.read()
    if not ret: break
    cv2.imwrite(dest + f'{i:0>5}.jpg', frame)
    


