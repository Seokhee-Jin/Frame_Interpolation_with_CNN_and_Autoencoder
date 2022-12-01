import cv2
import numpy as np
from tensorflow import keras
from train_networks import Networks
from make_clip_and_frames import video_to_frame, frame_to_video
import os
import shutil
from PIL import Image



_video_path = r"C:\Users\32174417\PycharmProjects\openSourceAI_22_2\video\sample.mp4"
_frame_weights = r"C:\Users\32174417\PycharmProjects\openSourceAI_22_2\train\1.5hrTransfer(해제)Naturalcheckpoints\transfer_middle_frame_predictor.h5"
_middle_frame_predictor_model = Networks.get_middle_frame_predictor()
_middle_frame_predictor_model.load_weights(_frame_weights)

def convert_video_double_fps_(video_path=_video_path, middle_frame_predictor_model=_middle_frame_predictor_model, original_fps=24):
    """
    훈련된 프레임 예측 모델을 사용해 원본 영상의 프레임을 두배로 늘려주는 함수이다.
    """
    #우선 임시폴더에 프레임 저장
    temp_dir = '__temp'
    video_to_frame(video_path, original_fps, temp_dir)

    #저장한 프레임을 다시 불러옴
    frame_dir = os.path.join(temp_dir, 'frame', os.path.splitext(os.path.basename(_video_path))[0])
    _frame_path_list = [os.path.join(frame_dir, frame) for frame in os.listdir(frame_dir) if frame.endswith('.jpg')]
    _frame_path_list.sort()
    frames_np = np.array([cv2.imread(frame) / 255.0 for frame in _frame_path_list]) # (1439, 100, 100, 3), 0~1 값

    first_frame = frames_np[:-1]
    second_frame = frames_np[1:]
    first_frame = np.expand_dims(first_frame, axis=0)  # (1, 1438, 100, 100, 3)
    second_frame = np.expand_dims(second_frame, axis=0)  # (1, 1438, 100, 100, 3)

    model_dataset = np.concatenate([first_frame, second_frame], axis=0) # (2, 1438, 100, 100, 3)

    model_dataset = model_dataset.swapaxes(0,1) #(1438, 2, 100, 100, 3)
    print(f'model_dataset.shape: {model_dataset.shape}')

    middle_frame = middle_frame_predictor_model.predict(model_dataset)
    print('prediction complete')
    print(f"middle_frame.shape: {middle_frame.shape}")

    middle_frame = middle_frame.reshape(model_dataset.shape[0], 100, 100, 3)  # (None, 1, 100, 100, 3) -> # (None, 100, 100, 3)
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
    frame_to_video(frame_dir, fps=original_fps*2, result_dir=result_dir)
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    convert_video_double_fps_()
