import network
from network import *
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from preprocess import *
import time


def callbacks(log_path, save_path, patience=5):
    def get_run_logdir():  # 실행할 때마다 하위 로그 디렉토리를 만들어 주기위한 경로 생성 함수.
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(log_path, run_id)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(save_path, save_best_only=True,
                                                    save_weights_only=True)  # 매 개선된 에폭마다 weights를 저장하기 위한 콜백
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=5) # n번의 에폭동안 val_loss가 개선되지 않으면 조기종료하는 콜백.
    tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())  # 로그를 저장하고 그림을 보기 위한 콜백

    return [checkpoint_cb, early_stopping_cb, tensorboard_cb]


def autoencoder_train(num_npy: int, epochs: int, iteration: int, batch_size: int=8 , save_path=r".\train\checkpoints\autoencoder_checkpoint.h5",
                      load_OK=True, load_path=r".\train\checkpoints\autoencoder_checkpoint.h5", log_path=r".\train\checkpoints\logs\autoencoder", npy_dataset_path=r"D:\data10sec\datasets"):
    '''
    오토인코더를 위한 훈련메소드. iteration을 하는 이유는 train dataset을 루프를 돌 때마다 새로 추출해서 훈련시키기 위함이다. (이미지가 커서 한번에 로드할 수 없었기에 차선방안.)
    '''

    auto_encoder = autoencoder()

    auto_encoder.compile(loss='mse', optimizer='adam', metrics=['acc'])
    if load_OK:
        auto_encoder.load_weights(load_path)

    for i in range(iteration):
        ndarray, dataset_size = load_npys_and_get_one_np(npy_dataset_path, num_npy=num_npy, num_seq=1)
        trainset, validset, testset = np_to_tf_dataset(ndarray, dataset_size)
        auto_encoder.fit(trainset, epochs=epochs, batch_size=batch_size, validation_data=validset,
                                  callbacks=callbacks(log_path, save_path)) # 위에서 정의한 모든 콜백 적용해서 피팅.
        auto_encoder.load_weights(load_path)
        del ndarray, trainset, validset, testset


def frame_model_train(num_npy: int, epochs: int, iteration: int, ae_load_path=r".\train\checkpoints\autoencoder_checkpoint.h5", batch_size: int = 8 , save_path=r".\train\checkpoints\frame_interpolation_model.h5",
                      load_OK=True, load_path=r".\train\checkpoints\frame_interpolation_model_checkpoint.h5", log_path=r".\train\checkpoints\logs\frame_interpolation_model", npy_dataset_path=r"D:\data10sec\datasets"):
    '''
   autoencoder_train 메소드를 개조. 2*100*100*3 크기의 프레임(앞 뒤 프레임)을 넣으면 1*100*100*3 크기의 단일 프레임(중간 프레임)을 반환.
    '''

    auto_encoder = autoencoder()
    auto_encoder.load_weights(ae_load_path)
    encoder = auto_encoder.layers[0]
    encoder.trainable = False # 오토 인코더는 이미 학습을 마쳤다고 가정.


    frame_model = middle_frame_prediction_model(5, ae_load_path=ae_load_path)

    frame_model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    if load_OK:
        frame_model.load_weights(load_path)

    for i in range(iteration):
        ndarray, dataset_size = load_npys_and_get_one_np(npy_dataset_path, num_npy=num_npy, num_seq=5)
        trainset, validset, testset = np_to_tf_dataset(ndarray, dataset_size, batch_size=8)
        frame_model.fit(trainset, epochs=epochs, batch_size=batch_size, validation_data=validset,
                        callbacks=callbacks(log_path, save_path, patience=10)) # 위에서 정의한 모든 콜백 적용해서 피팅.
        frame_model.load_weights(load_path)
        del ndarray, trainset, validset, testset


if __name__ == '__main__':
    gpu_limit()
    os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
    #autoencoder_train(6, 100, 1000)
    frame_model_train(3, 100, 100, load_OK=False)
