import subprocess as sp
from os import path
FALL_VIDEOS_COUNT = 30
ADL_VIDEOS_COUNT = 40

def download_and_unzip(video_url, downloaded_file, rename_to):
    sp.run(['wget', video_url], check=True)
    sp.run(['unzip', downloaded_file], check=True)
    sp.run(['rm', downloaded_file], check=True)
    sp.run(['mv', downloaded_file.split('.')[0], rename_to], check=True)
    
def download_fall_data():
    url_root = "fenix.univ.rzeszow.pl/~mkepski/ds/data/"
    url_name_template = "fall-{:0>2}-cam0-rgb.zip"
    rename_to_template = 'video ({})'
    for video_index in range(1, FALL_VIDEOS_COUNT + 1):
        file_name = url_name_template.format(video_index)
        url = url_root + file_name
        rename_to = rename_to_template.format(video_index)
        download_and_unzip(url, file_name, rename_to)
        
def download_adl_data():
    url_root = "fenix.univ.rzeszow.pl/~mkepski/ds/data/"
    url_name_template = "adl-{:0>2}-cam0-rgb.zip"
    rename_to_template = 'video ({})'
    for video_index in range(5, ADL_VIDEOS_COUNT + 1):
        file_name = url_name_template.format(video_index)
        url = url_root + file_name
        rename_to = rename_to_template.format(video_index)
        download_and_unzip(url, file_name, rename_to)
if __name__ == '__main__':
    download_adl_data()
