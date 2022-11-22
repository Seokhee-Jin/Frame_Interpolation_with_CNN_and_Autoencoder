#Python 3.8.15
#pip install tensorflow==2.11.0
#pip list --format=freeze > ./requirements.txt
import os
import cv2
import numpy as np
from tensorflow import keras
from keras.layers import *

import video_re


def MakingSampleVideoAndFrames(root_dir=r"D:\data", video_dir=r"D:\data\video"): # 샘플데이터 만들기 위한 임시 메소드. data_dir안에 video폴더 안에는 이미 동영상파일들이 있어야함.
    import os, cv2, video_re
    os.makedirs(root_dir, exist_ok=True)

    video_list = os.listdir(video_dir)
    for video in video_list:
        video_cv = cv2.VideoCapture(os.path.join(video_dir, video))
        end_sec = int((video_cv.get(cv2.CAP_PROP_FRAME_COUNT) / video_cv.get(cv2.CAP_PROP_FPS)))  # 동영상 길이
        end_sec = 180 # 지우기
        video_re.clip_in_directory(os.path.join(video_dir, video), end_sec, end_sec+600,
                                   os.path.splitext(video)[0], root_dir)

    clip_dir = os.path.join(root_dir, 'clip')
    clip_list = os.listdir(clip_dir)
    for clip in clip_list:
        video_re.video_to_frame_in_directory(os.path.join(clip_dir, clip), 24, root_dir)

    frame_folder_dir = os.path.join(root_dir, 'frame')
    frame_folder_list = os.listdir(frame_folder_dir)
    for frame_folder in frame_folder_list:
        video_re.frame_to_npy(os.path.join(frame_folder_dir, frame_folder), root_dir)

if __name__ == '__main__':
    MakingSampleVideoAndFrames(root_dir=r"D:\data10min", video_dir=r"D:\data\video")

r'''    np1 = np.load(r"C:\Users\32174417\PycharmProjects\opensourceAI_22_2\datasets\sample.npy") #npy(이미지가 행렬변환된 파일) 불러오기
    print('np1.shape:', np1.shape)

    cv2.imshow('imgWidow', np1[123]) # 행렬을 이미지로 보여주기.
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''








