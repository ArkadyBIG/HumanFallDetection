import shutil
from glob import glob
import os


for i, video_path in enumerate(glob('*/*/*.png')):
	if video_path.startswith('videos'):
		continue
	video_name = video_path.split('/')[-1].rsplit('_', 1)[0]
	frame_num = video_path.split('/')[-1].rsplit('_', 1)[-1].split('.')[0]
	video_dest = f'videos/{video_name}/'
	os.makedirs(video_dest, exist_ok=True)
	shutil.copy2(video_path, video_dest + f'{frame_num}.png')
	if i % 300 == 0:
		print(i)



