#pip list --format=freeze > ./requirements.txt

import cv2
import numpy as np
from tensorflow import keras
from train_networks import Networks
from make_clip_and_frames import video_to_frame, frame_to_video
import os
import shutil
from PIL import Image
import cv2
from moviepy.editor import *

_video_path = os.path.join("video", "Avatar130.mp4")
_frame_weights = os.path.join("train", "checkpoints", "transfer_middle_frame_predictor.h5")
_middle_frame_predictor_model = Networks.get_middle_frame_predictor()
_middle_frame_predictor_model.load_weights(_frame_weights)

def convert_video_double_fps_(video_filename, save_filename, middle_frame_predictor_model=_middle_frame_predictor_model,
                              original_fps=243):
    """
    훈련된 프레임 예측 모델을 사용해 원본 영상의 프레임을 두배로 늘려주는 함수이다.
    """
    # fps
    clip = VideoFileClip(video_filename)
    original_fps = clip.fps
    clip.close()

    # 우선 임시폴더에 프레임 저장
    temp_dir = '__temp'
    video_to_frame(video_filename, original_fps, temp_dir)
    frame_dir = os.path.join(temp_dir, "frame", os.path.splitext(os.path.basename(video_filename))[0])

    # 저장한 프레임을 다시 불러옴
    _frame_path_list = [os.path.join(frame_dir, frame) for frame in os.listdir(frame_dir) if frame.endswith('.jpg')]
    _frame_path_list.sort()
    frames_np = np.array([cv2.imread(frame) / 255.0 for frame in _frame_path_list])  # (1439, 100, 100, 3), 0~1 값

    print(frames_np.shape)

    first_frame = frames_np[:-1]
    second_frame = frames_np[1:]

    model_dataset = np.array([first_frame, second_frame])  # (2, 1438, 100, 100, 3)
    model_dataset = model_dataset.swapaxes(0, 1)  # (1438, 2, 100, 100, 3)
    print(f'model_dataset.shape: {model_dataset.shape}')

    middle_frame = middle_frame_predictor_model.predict(model_dataset)
    print('prediction complete')
    print(f"middle_frame.shape: {middle_frame.shape}")

    middle_frame = middle_frame.reshape(model_dataset.shape[0], 100, 100,
                                        3)  # (None, 1, 100, 100, 3) -> # (None, 100, 100, 3)
    middle_frame = (middle_frame * 255).astype(np.uint8)
    print(f"middle_frame.shape: {middle_frame.shape}")
    print(middle_frame.max(), middle_frame.min())
    print('prediction complete')

    # 예측한 중간 프레임을 폴더에 저장하자.
    for i, np_frame in enumerate(middle_frame):
        im = Image.fromarray(np_frame[:, :, ::-1])
        print(f"{os.path.join(frame_dir, '% 06d.m.jpg' % i)} is saved.")
        im.save(os.path.join(frame_dir, '%06d.m.jpg' % i))
        print(f"{os.path.join(frame_dir, '% 06d.m.jpg' % i)} is saved.")

    frame_to_video(frame_dir, save_filename, original_fps * 2)
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    convert_video_double_fps_("Sample_Avatar.mp4", "Sample_Avatar_x2.mp4")

