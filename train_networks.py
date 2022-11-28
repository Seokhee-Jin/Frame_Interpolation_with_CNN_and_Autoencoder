import make_dataset
import numpy as np
import os
import sys
from tensorflow import keras
import tensorflow as tf
import time

_npy_dataset_path = r"D:\data10sec\datasets"
_ae_model_load_path = os.path.join('train', 'checkpoints', 'proto_ae_123321')

_ae_weights_save_path = os.path.join('train', 'checkpoints', 'autoencoder_checkpoint.h5')
_ae_weights_load_path = _ae_weights_save_path
_ae_log_path = os.path.join('train', 'logs', 'autoencoder')

_frame_weights_save_path = os.path.join('train', 'checkpoints', 'middle_frame_predictor.h5')
_frame_weights_load_path = _frame_weights_save_path
_frame_log_path = os.path.join('train', 'logs', 'middle_frame_predictor')




class Networks:
    @staticmethod
    def get_pretrained_autoencoder(ae_model_load_path=_ae_model_load_path):
        """로드한 오토인코더 모델을 인코더와 디코더로 분리해서 반환"""
        autoencoder = keras.models.load_model(ae_model_load_path)
        len_half = len(autoencoder.layers) // 2

        encoder = keras.models.Sequential(autoencoder.layers[:len_half], name="encoder")
        decoder = keras.models.Sequential(autoencoder.layers[len_half:], name="decoder")

        return autoencoder, encoder, decoder

    @staticmethod
    def get_middle_frame_predictor(ae_model_load_path: str = _ae_model_load_path, ae_trainable: bool = False):
        """
        이 모델은 FindOptimalAutoencoder에서 이미 세팅된 오토인코더를 전이하여 구성한 모델이다.
        훈련과정에서 성능향상이 더딜 경우 ar_trainable = True로 바꿔서 훈련을 이어가자.
        :param ae_model_load_path: 세팅된 오토인코더가 저장된 경로.
        :param ae_trainable: 오토 인코더는 이미 학습을 마쳤다고 가정하기 때문에 False를 기본값으로 한다.
        :return: x
        """

        _, encoder, decoder = Networks.get_pretrained_autoencoder(ae_model_load_path=ae_model_load_path)
        encoder.trainable = ae_trainable
        decoder.trainable = ae_trainable

        def _residual_unit(tensor, filters: int):
            skip_value = keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(tensor)
            skip_value = keras.layers.BatchNormalization()(skip_value)
            ru_output = keras.layers.Activation('elu')(skip_value)
            ru_output = keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(ru_output)
            ru_output = keras.layers.BatchNormalization()(ru_output)
            added_output = keras.layers.Add()([skip_value, ru_output])
            return keras.layers.Activation('elu')(added_output)

        frames = keras.layers.Input(shape=(2, 100, 100, 3))  # 두 개 프레임에 각각 인코더를 적용해 차원을 축소한다.
        frame1 = keras.layers.Reshape((100, 100, 3), name="frame1")(frames[:, 0])  # 프레임1
        frame1 = encoder(frame1)  # (None, 50, 50, 4)

        frame3 = keras.layers.Reshape((100, 100, 3), name="frame3")(frames[:, 1])  # 프레임3
        frame3 = encoder(frame3)  # (None, 50, 50, 4)

        frame2 = keras.layers.concatenate((frame1, frame3),
                                          axis=-1)  # (None, 50, 50, 8): 인코딩된 두 프레임을 이어붙여서 8개의 피쳐맵으로 만든다.
        frame2 = _residual_unit(frame2, 64)
        frame2 = _residual_unit(frame2, 32)
        frame2 = _residual_unit(frame2, 16)
        frame2 = _residual_unit(frame2, 8)
        frame2 = _residual_unit(frame2, 4)  # (None, 50, 50, 4)
        frame2 = keras.layers.Conv2DTranspose(3, 5, strides=2, padding='same')(frame2)
#        frame2 = decoder(frame2)  # (None, 100, 100, 3)
        frame2 = keras.layers.Reshape((1, 100, 100, 3))(frame2)  # (None, 1, 100, 100, 3) 타겟 형태에 맞추기 위한 reshape

        model = keras.Model(inputs=frames, outputs=frame2, name="middle_frame_predictor")

        return model


class Train:
    @staticmethod
    def callbacks(log_path, weights_save_path, patience=5):
        """
        에폭마다 수행할 콜백들을 리스트로 반환하는 함수.
        터미널에 아래줄 입력하면 텐서보드 볼 수 있음 \n
        tensorboard --logdir=./drive/MyDrive/Colab_Notebooks/my_logs --port=6006
        """
        def get_run_logdir():  # 실행할 때마다 하위 로그 디렉토리를 만들어 주기위한 경로 생성 함수.
            run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
            return os.path.join(log_path, run_id)

        # 매 개선된 에폭마다 weights를 저장하기 위한 콜백
        checkpoint_cb = keras.callbacks.ModelCheckpoint(weights_save_path, save_best_only=True, save_weights_only=True)
        # n번의 에폭동안 val_loss가 개선되지 않으면 조기종료하는 콜백.
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
        # 로그를 저장하고 그림을 보기 위한 콜백
        tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())

        return [checkpoint_cb, early_stopping_cb, tensorboard_cb]

    @staticmethod
    def train_middle_frame_predictor(num_npy: int, epochs: int, iteration: int, ae_trainable: bool,
                                     batch_size: int = 32, patience=5,
                                     ae_model_load_path=_ae_model_load_path, weights_load_ok=True,
                                     weights_load_path=_frame_weights_load_path,
                                     weights_save_path=_frame_weights_save_path,
                                     log_path=_frame_log_path, npy_dataset_path=_npy_dataset_path):
        """
        한번의 iteration 동안 npy_dataset_path에 있는 npy들 중에서 num_npy개 만큼 랜덤으로 골라서 데이터 셋으로 가공하고 epochs만큼 훈련한다. \n
        이미 가중치가 세팅된 오토인코더를 로드하여 전이학습을 진행할 것이며, ae_trainable = True일 경우 낮은 학습률 하에서 가중치 갱신을 허용한다. \n

        :param num_npy: 랜덤으로 샘플링되어 데이터셋으로 가공될 npy의 갯수.
        :param epochs:
        :param iteration: 데이터셋을 만들고 모델을 훈련하는 반복문의 반복 횟수.
        :param ae_trainable: 전이된 오토인코더의 가중치 갱신 여부.
        :param batch_size:
        :param patience: 이 숫자만큼의 에폭동안 val_loss가 감소하지 않을 시 훈련을 조기 종료.
        :param ae_model_load_path: 이미 가중치가 세팅된 오토인코더 모델을 로드할 경로.
        :param weights_load_ok: 저장된 middle_frame_predictor의 가중치를 로드해서 훈련을 이어갈지 여부.
        :param weights_load_path:
        :param weights_save_path:
        :param log_path: 로그가 저장되는 경로. 텐서보드로 모니터링 가능.
        :param npy_dataset_path:
        :return: x
        """

        frame_model = Networks.get_middle_frame_predictor(ae_model_load_path=ae_model_load_path, ae_trainable=ae_trainable)

        frame_model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['acc'])
        if weights_load_ok:
            frame_model.load_weights(weights_load_path)

        print("You can monitor this training by using Tensor Board. Type the next line in terminal window")
        print(f"<tensorboard --logdir={log_path} --port=6006>")
        for i in range(iteration):
            print(f"{frame_model.name}: {i + 1}th iteration starts.")
            ndarray, dataset_size = make_dataset.load_npys_and_get_one_np(npy_dataset_path, num_npy=num_npy, num_seq=3)
            trainset, validset, testset = make_dataset.np_to_tf_dataset(ndarray, dataset_size, batch_size=32)
            frame_model.fit(trainset, epochs=epochs, batch_size=batch_size, validation_data=validset,
                            callbacks=Train.callbacks(log_path, weights_save_path, patience=patience))  # 위에서 정의한 모든 콜백 적용해서 피팅.
            del ndarray, trainset, validset, testset

class train_network:
    @staticmethod
    def gpu_limit():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    @staticmethod
    def gpu_limit2():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

if __name__ == '__main__':
    train_network.gpu_limit()
    os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
    Train().train_middle_frame_predictor(20, 100, 1000, ae_trainable=False, weights_load_ok=False)
    # if performance is not improving, stop previous line, and run next line.
    #Train().train_middle_frame_predictor(20, 100, 1000, ae_trainable=True, weights_load_ok=True)
    # Train().train_autoencoder(30, 30, 100, weights_load_ok=True) do not use. doesn't have any meaning
