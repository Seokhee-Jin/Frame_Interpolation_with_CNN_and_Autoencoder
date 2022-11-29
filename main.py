import cv2
import numpy as np
from tensorflow import keras
from train_networks import Networks

testmodel = Networks.get_middle_frame_predictor(ae_transfer_ok=True)
testmodel.layers
for layer in testmodel.layers:
    if layer.name == 'encoder' or layer.name == 'decoder':
        print(layer.name)
        #layer.trainable = as_traina

def convert_video_double_fps_(video_path, middle_frame_predictor_model, original_fps):


    video = cv2.VideoCapture(video_path)

    count = 0
    while(video.isOpened()):
        count += 1


    video.release()