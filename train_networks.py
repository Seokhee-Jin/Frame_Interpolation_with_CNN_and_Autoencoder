import make_dataset
import os
from tensorflow import keras
import tensorflow as tf
import time

_npy_dataset_path = r"D:\data10sec\datasets"
_ae_model_load_path = os.path.join('train', 'checkpoints', 'proto_ae_123321.h5')

_frame_model_save_path = os.path.join('train', 'checkpoints', 'transfer_middle_frame_predictor.h5')
_frame_model_load_path = _frame_model_save_path
_frame_log_path = os.path.join('train', 'logs', 'middle_frame_predictor')


class Networks:
    @staticmethod
    def get_pretrained_autoencoder(ae_model_load_path=_ae_model_load_path):
        """find_optimal_autoencoder.py에서 greedy 탐색법을 이용해 찾은 오토인코더를 가중치까지 그대로 복원해서 반환한다 \n
        :return autoencoder, encoder, decoder
        """
        autoencoder = keras.models.load_model(ae_model_load_path)
        len_half = len(autoencoder.layers) // 2

        encoder = keras.models.Sequential(autoencoder.layers[:len_half], name="encoder")
        decoder = keras.models.Sequential(autoencoder.layers[len_half:], name="decoder")

        return autoencoder, encoder, decoder


    @staticmethod
    def get_initial_autoencoder():
        """
        find_optimal_autoencoder.py에서 greedy 탐색법을 이용해 찾은 오토인코더의 구조를 재정의하였다.
        가중치가 초기값 그대로인 오토인코더 모델이 필요할 때 사용하는 메소드이다.\n
        :return: autoencoder, encoder, decoder
        """
        inputs = keras.layers.Input(shape=(100, 100, 3))  # 1번 오토인코더의 인코더와 디코더는 최종 조합된 오토인코더에서 가장 바깥 부분이 된다.
        encoded = keras.layers.Conv2D(9, 5, strides=2, padding='same', activation='elu')(inputs)
        encoded = keras.layers.Conv2D(9, 3, strides=1, padding='same', activation='elu')(encoded)
        encoded = keras.layers.Conv2D(5, 3, strides=1, padding='same', activation='elu')(encoded)
        encoded = keras.layers.Conv2D(5, 3, strides=1, padding='same', activation='elu')(encoded)
        encoded = keras.layers.Conv2D(4, 3, strides=1, padding='same', activation='elu')(encoded)
        encoded = keras.layers.Conv2D(4, 3, strides=1, padding='same', activation='elu')(encoded)
        encoder = keras.Model(inputs=inputs, outputs=encoded, name='proto_enc123')

        encoded_inputs = keras.layers.Input(shape=(50, 50, 4))
        decoded = keras.layers.Conv2DTranspose(5, 3, strides=1, padding='same', activation='elu')(encoded_inputs)
        decoded = keras.layers.Conv2DTranspose(5, 3, strides=1, padding='same', activation='elu')(decoded)
        decoded = keras.layers.Conv2DTranspose(9, 3, strides=1, padding='same', activation='elu')(decoded)
        decoded = keras.layers.Conv2DTranspose(9, 3, strides=1, padding='same', activation='elu')(decoded)
        decoded = keras.layers.Conv2DTranspose(9, 3, strides=1, padding='same', activation='elu')(decoded)
        decoded = keras.layers.Conv2DTranspose(9, 5, strides=2, padding='same', activation='elu')(decoded)
        decoded = keras.layers.Conv2D(3, 1)(decoded)
        decoder = keras.Model(inputs=encoded_inputs, outputs=decoded, name='proto_dec321')

        autoencoder = keras.models.Sequential([encoder, decoder], name='autoencoder_1233321')

        return autoencoder, encoder, decoder

    @staticmethod
    def get_middle_frame_predictor(ae_transfer_ok = True, ae_model_load_path: str = _ae_model_load_path):
        """
        ae_transfer_ok = True일 경우
        이 함수는 FindOptimalAutoencoder에서 이미 세팅된 오토인코더를 전이하여 구성된 모델을 반환한다..
        :param ae_model_load_path: 세팅된 오토인코더가 저장된 경로.
        :return: x
        """

        if ae_transfer_ok:
            _, encoder, decoder = Networks.get_pretrained_autoencoder(ae_model_load_path=ae_model_load_path)
        else:
            _, encoder, decoder = Networks.get_initial_autoencoder()

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
#        frame2 = keras.layers.Conv2DTranspose(3, 5, strides=2, padding='same')(frame2)
        frame2 = decoder(frame2)  # (None, 100, 100, 3)
        frame2 = keras.layers.Reshape((1, 100, 100, 3))(frame2)  # (None, 1, 100, 100, 3) 타겟 형태에 맞추기 위한 reshape

        middle_frame_predictor = keras.Model(inputs=frames, outputs=frame2, name="middle_frame_predictor")

        return middle_frame_predictor


class Train:
    @staticmethod
    def callbacks(log_path, weights_save_path, save_weights_only = True, patience=5):
        """
        에폭마다 수행할 콜백들을 리스트로 반환하는 함수.
        터미널에 아래줄 입력하면 텐서보드 볼 수 있음 \n
        tensorboard --logdir=./drive/MyDrive/Colab_Notebooks/my_logs --port=6006
        """
        def get_run_logdir():  # 실행할 때마다 하위 로그 디렉토리를 만들어 주기위한 경로 생성 함수.
            run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
            return os.path.join(log_path, run_id)

        # 매 개선된 에폭마다 weights를 저장하기 위한 콜백
        checkpoint_cb = keras.callbacks.ModelCheckpoint(weights_save_path, save_best_only=True, save_weights_only=save_weights_only)
        # n번의 에폭동안 val_loss가 개선되지 않으면 조기종료하는 콜백.
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
        # 로그를 저장하고 그림을 보기 위한 콜백
        tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())

        return [checkpoint_cb, early_stopping_cb, tensorboard_cb]


    @staticmethod
    def train_middle_frame_predictor(num_npy: int, epochs: int, iteration: int, batch_size: int = 32,
                                     ae_transfer_ok = True, ae_transfer_trainable: bool = True, patience_of_early_stopping=5, resume_train=True):
        """
        한번의 iteration 동안 npy_dataset_path에 있는 npy들 중에서 num_npy개 만큼 랜덤으로 골라서 데이터 셋으로 가공하고 epochs만큼 훈련한다. \n
        이미 가중치가 세팅된 오토인코더를 로드하여 전이학습을 진행할 것이며, ae_trainable = True일 경우 낮은 학습률 하에서 가중치 갱신을 허용한다. \n

        :param num_npy: 랜덤으로 샘플링되어 데이터셋으로 가공될 npy의 갯수.
        :param epochs:
        :param iteration: 데이터셋을 만들고 모델을 훈련하는 반복문의 반복 횟수.
        :param ae_transfer_trainable: 전이된 오토인코더의 가중치 갱신 여부.
        :param batch_size:
        :param patience_of_early_stopping: 이 숫자만큼의 에폭동안 val_loss가 감소하지 않을 시 훈련을 조기 종료.
        :param resume_train: 저장된 middle_frame_predictor의 가중치를 로드해서 임의 중단된 훈련을 재개할지 여부.
        :return: x
        """
        frame_model = Networks.get_middle_frame_predictor(ae_transfer_ok=ae_transfer_ok,
                                                          ae_model_load_path=_ae_model_load_path)

        # 중단된 훈련을 재개할 경우 중간 저장된 모델을 로드.
        if resume_train:
            frame_model.load_weights(_frame_model_save_path)

        # 전이된 층을 동결하거나 활성화함.
        for layer in frame_model.layers:
            if layer.name == 'encoder' or layer.name == 'decoder':
                layer.trainable = ae_transfer_trainable

        frame_model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['acc'])


        print("\n\nYou can monitor this training by using Tensor Board. Type the next line in terminal window.")
        print(f": tensorboard --logdir={_frame_log_path} --port=6006\n\n")
        start_time = time.strftime('%y/%m/%d %H:%M', time.localtime())
        for i in range(iteration):
            current_time = time.strftime('%y/%m/%d %H:%M', time.localtime())
            print("--------------------------")
            print(f"Start local time : {start_time}")
            print(f"Current local time : {current_time}")
            print(f"{frame_model.name}: {i + 1}th iteration starts.")
            # 모델간의 비교를 위해서 시드를 정해두자.
            ndarray, dataset_size = make_dataset.load_npys_and_get_one_np(_npy_dataset_path, num_npy=num_npy, num_seq=3, seed=i*i)
            trainset, validset, testset = make_dataset.np_to_tf_dataset(ndarray, dataset_size, batch_size=32, seed=i*i)
            #save_weights_only=False로 했다가 갖은 고생을 했다.. 로드할때 번거롭더라도 True로 쓰자.
            frame_model.fit(trainset, epochs=epochs, batch_size=batch_size, validation_data=validset,
                            callbacks=Train.callbacks(_frame_log_path, _frame_model_save_path, save_weights_only=True, patience=patience_of_early_stopping))  # 위에서 정의한 모든 콜백 적용해서 피팅.
            del ndarray, trainset, validset, testset

class GPU_limit:
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
    GPU_limit.gpu_limit()
    os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')

    """Res Unit을 사용하지 않은 사전훈련된 오토인코더를 동결해서 전이 학습해보기."""
    # 처음 훈련시
    Train.train_middle_frame_predictor(20, 100, 1000, ae_transfer_ok=True, ae_transfer_trainable=False, resume_train=False)
    # if performance is not improving, stop previous line, and run next line.
    #Train().train_middle_frame_predictor(20, 100, 1000, ae_transfer_ok=True, ae_transfer_trainable=True, resume_train=True)


    """Res Unit을 사용하지 않은 초기 상태의 오토인코더를 사용하여 학습해보기"""
    #처음 훈련시
    #Train.train_middle_frame_predictor(20, 100, 10000, ae_transfer_ok=False, ae_transfer_trainable=True, resume_train=False)
    #이어서 훈련시
    #Train.train_middle_frame_predictor(20, 100, 10000, ae_transfer_ok=False, ae_transfer_trainable=True, resume_train=True)