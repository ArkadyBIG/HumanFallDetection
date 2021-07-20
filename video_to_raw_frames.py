import cv2
import os
path = 'videos/human_crawling.mp4'
name = path.split('.')[0].split('/')[-1]
dest = f'raw_frames/{name}/'

os.makedirs(dest, exist_ok=True)
cap = cv2.VideoCapture(path)

for i in range(1_000_000):
    ret, frame = cap.read()
    if not ret: break
    cv2.imwrite(dest + f'{i:0>5}.jpg', frame)
    


