from moviepy.editor import *
import moviepy.video.fx.all as vfx
import os
import cv2
import numpy as np

_root_dir_for_saving_clips_and_frames = r"D:\data10sec"
_original_video_dir = r"D:\data\video"

def clip_video(filename: str, t_start, t_end, clip_name: str, directory: str):
    '''clip between times t_start and t_end, which can be expressed in seconds (15.35),
        in (min, sec), in (hour, min, sec), or as a string: '01:03:05.35'.'''

    extension = os.path.splitext(filename)[1]
    clip = VideoFileClip(filename)
#   clip = clip.without_audio()
    (w, h) = clip.size
    # 높이는 그대로 하고 너비를 잘라서 정사각 비디오로 만들기 ex: 1280*720 -> 720*720
    clip = vfx.crop(clip, width=h, height=h, x_center=w / 2,
                    y_center=h / 2)
    clip = vfx.resize(clip, width=100, height=100)
    clip = clip.subclip(t_start, t_end)

    clip_directory = os.path.join(directory, 'clip')
    if not os.path.exists(clip_directory):
        os.mkdir(clip_directory)
    clip.write_videofile(os.path.join(clip_directory, clip_name + extension))


def video_to_frame(filename, H_fps, directory):  # 사용자가 원하는 프레임으로 캡쳐. 원하는 장소에 저장.
    """
    :param filename: 동영상 경로
    :param H_fps: 초당 얼마나 많은 이미지를 저장할 것인지.
    :param directory: 캡쳐한 이미지들을 저장할 디렉토리
    """

    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]
    directory_path = os.path.join(directory, 'frame', filename_without_extension)

    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
    except OSError:
        print('Error: Creating directory. ' + directory_path)

    video = cv2.VideoCapture(filename)

    if not video.isOpened():
        print("Could not Open :", filename)
        exit(0)

    fps = round(video.get(cv2.CAP_PROP_FPS))
    HPS = round(fps / H_fps)

    count = 0
    number = 0

    while (True):

        ret, image = video.read()
        if not ret: break
        if (count % HPS == 0):
            cv2.imwrite(os.path.join(directory_path, "%06d.jpg" % number), image)
            print('Saved frame number :', filename_without_extension, str(int(video.get(1))))
            number += 1
        count += 1

    video.release()

def frame_to_video(frame_dir, fps, result_dir='.'):
    """
    :param frame_dir: 비디오로 변환할 프레임이 있는 디렉토리 이름
    :param result_dir: 변환 결과 비디오 이름 지정
    :param fps: fps 지정
    """
    clips = []

    os.path.basename(frame_dir)
    frame_dir_list = sorted(os.listdir(frame_dir))

    for filename in frame_dir_list:
        if filename.endswith(".jpg"):
            clips.append(ImageClip(os.path.join(frame_dir, filename)).set_duration(1/fps))

    video = concatenate_videoclips(clips, method="compose")
    video.write_videofile(os.path.join(result_dir, os.path.basename(frame_dir) + '.mp4'), fps=fps)


def frame_to_npy(frames_dir: str, directory: str):  # frames를 (batch_size, height, width, channel)의 numpy 배열로 저장하기.
    dataName = os.path.split(frames_dir)[1]
    frames = [img for img in os.listdir(frames_dir) if img.endswith(".jpg")]  # jpg만 모아서 리스트에 저장.
    frames.sort()

    frames_array = []
    for img in frames:
        frames_array.append(cv2.imread(os.path.join(frames_dir, img)))
        print(img + " appended to {}.npy".format(dataName))

    frames_np = np.array(frames_array)
    print(frames_np.shape)

    directory_datasets = os.path.join(directory, 'datasets')
    if not os.path.exists(directory_datasets):
        os.mkdir(directory_datasets)

    np.save(os.path.join(directory_datasets, dataName), frames_np)


def make_clips_and_frames(root_dir=r"D:\data", video_dir=r"D:\data\video"):
    """
    # 샘플데이터 만들기 위한 임시 메소드. data_dir안에 video폴더 안에는 이미 동영상파일들이 있어야함.
    """

    os.makedirs(root_dir, exist_ok=True)

    video_list = os.listdir(video_dir)
    for video in video_list:
        video_cv = cv2.VideoCapture(os.path.join(video_dir, video))
        end_sec = int((video_cv.get(cv2.CAP_PROP_FRAME_COUNT) / video_cv.get(cv2.CAP_PROP_FPS)))  # 동영상 길이
        for tensec in range(int(end_sec / 10)):  # 십초 단위로 쪼갠다..
            clip_video(os.path.join(video_dir, video), tensec * 10, tensec * 10 + 10,
                       os.path.splitext(video)[0] + "%04d" % tensec, root_dir)

    clip_dir = os.path.join(root_dir, 'clip')
    clip_list = os.listdir(clip_dir)
    for clip in clip_list:
        video_to_frame(os.path.join(clip_dir, clip), 24, root_dir)

    frame_folder_dir = os.path.join(root_dir, 'frame')
    frame_folder_list = os.listdir(frame_folder_dir)
    for frame_folder in frame_folder_list:
        frame_to_npy(os.path.join(frame_folder_dir, frame_folder), root_dir)


if __name__ == '__main__':
    frame_to_video(r"D:\data10sec\frame\Avatar0001", 24)
    #make_clips_and_frames(root_dir=_root_dir_for_saving_clips_and_frames, video_dir=_original_video_dir)

