from tensorflow import keras
from train_networks import Networks
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

_imagefile1 = r'D:\data10sec\frame\Avatar0001\000072.jpg'
_imagefile2 = r'D:\data10sec\frame\Avatar0130\000072.jpg'
_ae_model_load_path = os.path.join('train', 'checkpoints', 'proto_ae_123321.h5')

_frame_weights_freeze = r"C:\Users\32174417\PycharmProjects\openSourceAI_22_2\train\오토인코더동결전이학습checkpoints\middle_frame_predictor.h5"
_frame_weights_natural = r"C:\Users\32174417\PycharmProjects\openSourceAI_22_2\train\동결x순수모델학습checkpoints\middle_frame_predictor.h5"

_frames_dir = r"D:\data10sec\frame\Avatar0001"
_frame_path_list = [os.path.join(_frames_dir, frame) for frame in os.listdir(_frames_dir) if frame.endswith('.jpg')]
_frame_path_list.sort()



def test_autoencoder(autoencoder_model):

    img1 = cv2.imread(_imagefile1)

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img1[:, :, ::-1])
    plt.title('result 1')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(np.reshape(autoencoder_model.predict(np.expand_dims(img1, axis=0) / 255.0), (100, 100, 3))[:, :, ::-1])
    plt.title('result 2')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def test_middle_frame_predictor(middle_frame_predictor_model):
    frames_np = np.array([cv2.imread(frame) / 255.0 for frame in _frame_path_list])
    duplicated_frames_np = np.empty((0, 100, 100, 3))  #
    for i in range(len(frames_np)):  # [1,2,3] -> [1,1,2,2,3,3]
        duplicated_frames_np = np.concatenate((duplicated_frames_np, np.expand_dims(frames_np[i], axis=0)), axis=0)
        duplicated_frames_np = np.concatenate((duplicated_frames_np, np.expand_dims(frames_np[i], axis=0)), axis=0)

    duplicated_frames_np = duplicated_frames_np[1:-1]  # [1,1,2,2,3,3]- > [1,2,2,3] # 맨 앞과 뒤를 제거.
    duplicated_frames_np = np.reshape(duplicated_frames_np,
                                ((len(duplicated_frames_np) // 2, 2, 100, 100, 3)))  # [1,2,2,3] -> [[1,2],[2,3]]

    num_frame_bunch = len(duplicated_frames_np)

    middle_frames = middle_frame_predictor_model.predict(duplicated_frames_np)
    middle_frames = np.reshape(middle_frames, (num_frame_bunch, 100, 100, 3))  # (None, 1, 100, 100, 3) -> # (None, 100, 100, 3)

    fig = plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.subplot(2, 4, i+1)
        plt.imshow(frames_np[i][:, :, ::-1])
        plt.title('original frame')
        plt.xticks([])
        plt.yticks([])

    for i in range(4):
        plt.subplot(2, 4, i+5)
        plt.imshow(middle_frames[i][:, :, ::-1])
        plt.title('middle frame')
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    autoencoder = keras.models.load_model(_ae_model_load_path)
    #test_autoencoder(autoencoder)

    #동결전이모델 테스트
    # _middle_frame_predictor_model = Networks.get_middle_frame_predictor()
    # _middle_frame_predictor_model.load_weights(_frame_weights_freeze)
    # test_middle_frame_predictor(_middle_frame_predictor_model)

    #전이x모델 테스트
    _middle_frame_predictor_model = Networks.get_middle_frame_predictor()
    _middle_frame_predictor_model.load_weights(_frame_weights_natural)
    test_middle_frame_predictor(_middle_frame_predictor_model)