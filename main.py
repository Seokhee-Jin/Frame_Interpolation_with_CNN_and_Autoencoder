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

# r"D:\data1min_\clip\Avatar130.mp4"
# r"D:\data1min_\clip\Singin'.In.The.Rain057.mp4"
# r"D:\data1min_\clip\Spirited.Away012.mp4"
# r"D:\data1min_\clip\Star.Wars091.mp4"
# r"D:\data1min_\clip\The.Lion.King004.mp4"

_video_path = "D:\data1min_\clip\The.Lion.King004_2.mp4"
_frame_weights = os.path.join("train", "checkpoints", "transfer_middle_frame_predictor.h5")
_middle_frame_predictor_model = Networks.get_middle_frame_predictor()
_middle_frame_predictor_model.load_weights(_frame_weights)


def convert_video_double_fps_(video_path=_video_path, middle_frame_predictor_model=_middle_frame_predictor_model,
                              original_fps=47.96):
    """
    훈련된 프레임 예측 모델을 사용해 원본 영상의 프레임을 두배로 늘려주는 함수이다.
    """

    # 우선 임시폴더에 프레임 저장
    temp_dir = '__temp'
    video_to_frame(video_path, original_fps, temp_dir)

    # 저장한 프레임을 다시 불러옴
    frame_dir = os.path.join(temp_dir, 'frame', os.path.splitext(os.path.basename(_video_path))[0])
    _frame_path_list = [os.path.join(frame_dir, frame) for frame in os.listdir(frame_dir) if frame.endswith('.jpg')]
    _frame_path_list.sort()
    frames_np = np.array([cv2.imread(frame) / 255.0 for frame in _frame_path_list])  # (1439, 100, 100, 3), 0~1 값

    first_frame = frames_np[:-1]
    second_frame = frames_np[1:]
    first_frame = np.expand_dims(first_frame, axis=0)  # (1, 1438, 100, 100, 3)
    second_frame = np.expand_dims(second_frame, axis=0)  # (1, 1438, 100, 100, 3)

    model_dataset = np.concatenate([first_frame, second_frame], axis=0)  # (2, 1438, 100, 100, 3)

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
    os.makedirs(os.path.join(os.path.split(_video_path)[0], 'double_FPS'), exist_ok=True)
    result_dir = os.path.join(os.path.split(_video_path)[0], 'double_FPS')
    frame_to_video(frame_dir, fps=original_fps * 2, result_dir=result_dir)
    shutil.rmtree(temp_dir)


def tempmethod(video_filename: str, save_filename: str, middle_frame_predictor_model=_middle_frame_predictor_model):
    video_clip = VideoFileClip(video_filename)
    original_audio = video_clip.audio
    original_duration = video_clip.duration
    original_fps = video_clip.fps
    # video_clip.get_frame()
    num_total_frames = int(original_duration * original_fps)

    original_frame = [video_clip.get_frame(i/original_fps)/255.0 for i in range(num_total_frames)]
    original_frame = np.array(original_frame)
    #   original_frame.shape: (1439, 100, 100, 3)

    first_frame = original_frame[:-1]
    second_frame = original_frame[1:]

    model_dataset = np.array([first_frame, second_frame]) # (2, 1438, 100, 100, 3)
    model_dataset = model_dataset.swapaxes(0, 1)  # (1438, 2, 100, 100, 3)
    print(f'model_dataset.shape: {model_dataset.shape}')

    middle_frame = middle_frame_predictor_model.predict(model_dataset)
    print('prediction complete')
    print(f"middle_frame.shape: {middle_frame.shape}")

    first_frame = original_frame[:-1]
    second_frame = original_frame[1:]
    first_frame = np.expand_dims(first_frame, axis=0)  # (1, 1438, 100, 100, 3)
    second_frame = np.expand_dims(second_frame, axis=0)  # (1, 1438, 100, 100, 3)

    model_dataset = np.concatenate([first_frame, second_frame], axis=0)  # (2, 1438, 100, 100, 3)

    model_dataset = model_dataset.swapaxes(0, 1)  # (1438, 2, 100, 100, 3)
    print(f'model_dataset.shape: {model_dataset.shape}')

    middle_frame = middle_frame_predictor_model.predict(model_dataset) # (1438, 1, 100, 100, 3)
    print('prediction complete')
    print(f"middle_frame.shape: {middle_frame.shape}")

    middle_frame = np.concatenate([middle_frame, np.zeros((1, 1, 100, 100, 3))], axis=0)
    print("!!",middle_frame.shape)
    #   (1438, 1, 100, 100, 3) -> (1439, 1, 100, 100, 3)
    #   : 프레임을 정확히 두배로 늘리려면 마지막에 검은색 프레임 한개를 채워줘야 함.
    middle_frame = middle_frame.reshape(middle_frame.shape[0], 100, 100, 3)
    #   (1439, 1, 100, 100, 3) -> (1439, 100, 100, 3)
    double_frame = np.array([original_frame, middle_frame])
    double_frame = double_frame.reshape((double_frame.shape[1]*2, 100, 100, 3))
    print(double_frame.shape)

    try:
        new_clip = VideoClip(lambda t: double_frame[int(t * original_fps)]*255, duration=original_duration)
        new_clip.write_videofile("testest.mp4", fps=original_fps * 2)
    except IndexError:
        exit(0)


if __name__ == "__main__":
    # convert_video_double_fps_()
    tempmethod(r"D:\data1min_\clip\Avatar130.mp4", "test.mp4")
