#Python 3.8.15
#pip install tensorflow==2.11.0
#pip list --format=freeze > ./requirements.txt
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def MakingSampleVideoAndFrames(filename: str, t_start, t_end, clip_name: str):
    from video_re import clip, video_to_frame, frame_to_npy
    extension = os.path.splitext(filename)[1]
    clip(filename, t_start, t_end, clip_name)
    print('clip created!')
    #video_to_frame("./video/" + clip_name + os.path.splitext(filename)[1], 24)
    video_to_frame(os.path.join(os.getcwd(), 'video', clip_name + extension), 24)
    print('frames created!')
    frame_to_npy(os.path.join(os.getcwd(), 'frame', clip_name))
    print('npy created')

MakingSampleVideoAndFrames(r"C:\Users\32174417\Downloads\The Lion King (1994)\The.Lion.King.1994.BluRay.720p.x264.YIFY.mp4", (0,10,0), (0,11,0), 'sample')

np1 = np.load(r"C:\Users\32174417\PycharmProjects\opensourceAI_22_2\datasets\sample.npy")
np1.shape

cv2.imshow('imgWidow', np1[123])
cv2.waitKey(0)
cv2.destroyAllWindows()