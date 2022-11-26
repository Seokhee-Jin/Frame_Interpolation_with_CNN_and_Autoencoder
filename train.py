import network
import preprocess
import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import time

_npy_dataset_path = r".\datasets"
_ae_save_path = r".\train\checkpoints\autoencoder_checkpoint.h5"
_ae_log_path = r".\train\logs\autoencoder"
_frame_save_path = r".\train\checkpoints\frame_interpolation_model.tf"
_frame_log_path = r".\train\logs\frame_interpolation_model"


def callbacks(log_path, save_path, patience=5):
    '''
    터미널에 아래줄 입력하면 텐서보드 볼 수 있음
    tensorboard --logdir=./drive/MyDrive/Colab_Notebooks/my_logs --port=6006
    '''
    def get_run_logdir():  # 실행할 때마다 하위 로그 디렉토리를 만들어 주기위한 경로 생성 함수.
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(log_path, run_id)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(save_path, save_best_only=True,
                                                    save_weights_only=True)  # 매 개선된 에폭마다 weights를 저장하기 위한 콜백
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True) # n번의 에폭동안 val_loss가 개선되지 않으면 조기종료하는 콜백.
    tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())  # 로그를 저장하고 그림을 보기 위한 콜백

    return [checkpoint_cb, early_stopping_cb, tensorboard_cb]



def autoencoder_train(num_npy: int, epochs: int, iteration: int, batch_size: int=8, save_path=_ae_save_path,
                      load_OK=True, load_path=_ae_save_path, log_path=_ae_log_path, npy_dataset_path=_npy_dataset_path):
    '''
    오토인코더를 위한 훈련메소드. iteration을 하는 이유는 루프를 돌 때마다 새로운 train datase를 샘플링하고 훈련시키기 위함이다. (이미지가 커서 데이터 세트를 한번에 로드할 수 없었기에 차선방안.)
    '''

    auto_encoder = keras.models.load_model(r".\train\checkpoints\proto_ae_123321")

   # auto_encoder.compile(loss='mse', optimizer='adam', metrics=['acc'])
    if load_OK:
        auto_encoder.load_weights(load_path)

    for i in range(iteration):
        print(f"{i + 1}th iteration starts.")
        ndarray, dataset_size = preprocess.load_npys_and_get_one_np(npy_dataset_path, num_npy=num_npy, num_seq=1)
        trainset, validset, testset = preprocess.np_to_tf_dataset(ndarray, dataset_size)
        auto_encoder.fit(trainset, epochs=epochs, batch_size=batch_size, validation_data=validset,
                                  callbacks=callbacks(log_path, save_path)) # 위에서 정의한 모든 콜백 적용해서 피팅.
        del ndarray, trainset, validset, testset


def frame_model_train(num_npy: int, epochs: int, iteration: int, ae_load_path=_ae_save_path, batch_size: int = 32 , save_path=_frame_save_path,
                      load_OK=True, load_path=_frame_save_path, log_path=_frame_log_path, npy_dataset_path=_npy_dataset_path):
    '''
   autoencoder_train 메소드를 개조. 5*100*100*3 크기의 프레임(앞 뒤 프레임)을 넣으면 1*100*100*3 크기의 단일 프레임(중간 프레임)을 반환.
    '''

    frame_model = network.middle_frame_prediction_model(5, ae_load_path=ae_load_path)

    frame_model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    if load_OK:
        frame_model.load_weights(load_path)

    for i in range(iteration):
        print(f"{i+1}th iteration starts.")
        ndarray, dataset_size = preprocess.load_npys_and_get_one_np(npy_dataset_path, num_npy=num_npy, num_seq=5)
        trainset, validset, testset = preprocess.np_to_tf_dataset(ndarray, dataset_size, batch_size=8)
        frame_model.fit(trainset, epochs=epochs, batch_size=batch_size, validation_data=validset,
                        callbacks=callbacks(log_path, save_path, patience=5)) # 위에서 정의한 모든 콜백 적용해서 피팅.
        del ndarray, trainset, validset, testset


if __name__ == '__main__':
    network.gpu_limit()
    os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
    frame_model_train(6, 30, 100, load_OK=False)
    #autoencoder_train(30, 30, 100, load_OK=True)
