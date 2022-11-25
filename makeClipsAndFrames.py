# Python 3.8.15
# pip install tensorflow==2.11.0
# pip list --format=freeze > ./requirements.txt
import os, cv2, video_re

def make_clips_and_frames(root_dir=r"D:\data",
                          video_dir=r"D:\data\video"):  # 샘플데이터 만들기 위한 임시 메소드. data_dir안에 video폴더 안에는 이미 동영상파일들이 있어야함.

    os.makedirs(root_dir, exist_ok=True)

    video_list = os.listdir(video_dir)
    for video in video_list:
        video_cv = cv2.VideoCapture(os.path.join(video_dir, video))
        end_sec = int((video_cv.get(cv2.CAP_PROP_FRAME_COUNT) / video_cv.get(cv2.CAP_PROP_FPS)))  # 동영상 길이
        for tensec in range(int(end_sec / 10)):  # 십초 단위로 쪼갠다..
            video_re.clip_in_directory(os.path.join(video_dir, video), tensec * 10, tensec * 10 + 10,
                                       os.path.splitext(video)[0] + "%04d" % tensec, root_dir)

    clip_dir = os.path.join(root_dir, 'clip')
    clip_list = os.listdir(clip_dir)
    for clip in clip_list:
        video_re.video_to_frame_in_directory(os.path.join(clip_dir, clip), 24, root_dir)

    frame_folder_dir = os.path.join(root_dir, 'frame')
    frame_folder_list = os.listdir(frame_folder_dir)
    for frame_folder in frame_folder_list:
        video_re.frame_to_npy(os.path.join(frame_folder_dir, frame_folder), root_dir)


if __name__ == '__main__':
    make_clips_and_frames(root_dir=r"D:\data10sec", video_dir=r"D:\data\video")
