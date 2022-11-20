#Python 3.8.15
#pip install tensorflow==2.11.0
#pip list --format=freeze > ./requirements.txt
import os
import cv2
import numpy as np


def MakingSampleVideoAndFrames(filename: str, t_start, t_end, clip_name: str): # 샘플데이터 만들기 위한 임시 메소드
    from video_re import clip, video_to_frame, frame_to_npy
    extension = os.path.splitext(filename)[1]
    clip(filename, t_start, t_end, clip_name)
    print('clip created!')
    video_to_frame(os.path.join(os.getcwd(), 'video', clip_name + extension), 24) # 라이온킹은 24프레임이다.
    print('frames created!')
    frame_to_npy(os.path.join(os.getcwd(), 'frame', clip_name))
    print('npy created')

if __name__ == '__main__':
    r'''MakingSampleVideoAndFrames(
        r"C:\Users\32174417\Downloads\The Lion King (1994)\The.Lion.King.1994.BluRay.720p.x264.YIFY.mp4", (0, 10, 0),
        (0, 11, 0), 'sample')''' # 라이온킹 파일의 10분 ~ 11분 부분을 자르고 프레임 이미지들로 저장하기. 프레임 이미지들은 행렬형태의 npy파일로 다시 변환해서 저장.

    np1 = np.load(r"C:\Users\32174417\PycharmProjects\opensourceAI_22_2\datasets\sample.npy") #npy(이미지가 행렬변환된 파일) 불러오기
    print('np1.shape:', np1.shape)

    cv2.imshow('imgWidow', np1[123]) # 행렬을 이미지로 보여주기.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


