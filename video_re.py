from moviepy.editor import *
import moviepy.video.fx.all as vfx
import os
import cv2
import numpy as np

# 영상 구간 자르기 & crop
def clip(filename, t_start, t_end, clip_name): #ori_name을 filename으로 수정
    '''clip between times t_start and t_end, which can be expressed in seconds (15.35),
        in (min, sec), in (hour, min, sec), or as a string: '01:03:05.35'.'''

    extension = os.path.splitext(filename)[1] # 확장자명
    clip = VideoFileClip(filename) # mp4가 아닌 다른 파일이 들어올 수도 있으므로 수정함.
    clip = clip.without_audio()
    (w, h) = clip.size
    clip = vfx.crop(clip, width=h, height=h, x_center=w / 2, y_center=h / 2) # 높이는 그대로. 너비를 잘라서 줄임.. ex: 1280*720 -> 720*720
    clip = vfx.resize(clip, width=100, height=100) # 압축. ex 720*720 -> 100*100
    clip = clip.subclip(t_start, t_end)
    if not os.path.exists('./video'):
        os.mkdir('./video')
    clip.write_videofile("./video/" + clip_name + extension)


# 영상 frame 분할

def video_to_frame(filename, H_fps): #사용자가 원하는 프레임으로 캡쳐

    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]
    directory_path = "./frame/" + filename_without_extension

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
            #h, w, c = image.shape
            #mid_x, mid_y = w // 2, h // 2
            #image = image[mid_y - 50:mid_y + 50, mid_x - 50:mid_x + 50]
            cv2.imwrite(directory_path + "//%06d.jpg" % number, image)
            print('Saved frame number :', str(int(video.get(1))))
            number += 1
        count += 1

    video.release()

def frame_to_npy(folderOfFrames): # frames를 (batch_size, height, width, channel)의 numpy 배열로 저장하기.
    dataName = os.path.split(folderOfFrames)[1]
    frames = [img for img in os.listdir(folderOfFrames) if img.endswith(".jpg")] #jpg만 모아서 리스트에 저장.
    frames.sort()

    frames_array = []
    for img in frames:
        frames_array.append(cv2.imread(os.path.join(folderOfFrames, img)))
        print(img + " appended")

    frames_np = np.array(frames_array)
    print(frames_np.shape)

    if not os.path.exists('./datasets'):
        os.mkdir('./datasets')

    output_dir = os.path.join(os.getcwd(), 'datasets')
    np.save(os.path.join(output_dir, dataName), frames_np)


if __name__ == "__main__":
    #clip("Clouds", 0, 10, "clip")
    video_to_frame("Clouds", 5)
