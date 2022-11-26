# pip install --user tensorflow_gpu==2.11.0  # 그래픽카드 적용

import tensorflow as tf
from tensorflow import keras


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


def return_autoencoder_and_encoder_and_decoder(autoencoder_path: str = r".\train\checkpoints\proto_ae_123321"):
    '''로드한 오토인코더 모델을 인코더와 디코더로 분리해서 반환'''
    autoencoder = keras.models.load_model(autoencoder_path)
    len_half = len(autoencoder.layers) // 2

    encoder = keras.models.Sequential(autoencoder.layers[:len_half], name='encoder')
    decoder = keras.models.Sequential(autoencoder.layers[len_half:], name='decoder')

    return autoencoder, encoder, decoder


def middle_frame_prediction_model(num_seq: int, ae_load_path: str = r".\train\checkpoints\autoencoder_checkpoint.h5"):
    '''

    :param ae_load_path: 오토 인코더를 로드한다. 인코더를 이용해 입력데이터의 차원을 줄인다.
    :return: 예측된 가운데 프레임이 반환된다.
    '''
    autoencoder, encoder, decoder = return_autoencoder_and_encoder_and_decoder()
    autoencoder.trainable = False; encoder.trainable = False; decoder.trainable = False
        # 오토 인코더는 이미 학습을 마쳤다고 가정. 따라서 가중치 갱신 중지

    frames = keras.layers.Input(shape=(4, 100, 100, 3))  # 네개 프레임에 각각 인코더를 적용해 차원을 축소한다.
    frame1 = keras.layers.Reshape((100, 100, 3), name='frame1')(frames[:, 0])  # 프레임1
    frame1 = encoder(frame1)  # (None, 50, 50, 4)
    frame1 = keras.layers.Reshape((1, 10000))(frame1)

    frame2 = keras.layers.Reshape((100, 100, 3), name='frame2')(frames[:, 1])  # 프레임2
    frame2 = encoder(frame2)
    frame2 = keras.layers.Reshape((1, 10000))(frame2)

    frame4 = keras.layers.Reshape((100, 100, 3), name='frame4')(frames[:, 2])  # 프레임4
    frame4 = encoder(frame4)
    frame4 = keras.layers.Reshape((1, 10000))(frame4)

    frame5 = keras.layers.Reshape((100, 100, 3), name='frame5')(frames[:, 3])  # 프레임5
    frame5 = encoder(frame5)
    frame5 = keras.layers.Reshape((1, 10000))(frame5)

    frame3 = keras.layers.concatenate([frame1, frame2, frame4, frame5], axis=1)
        # (None, 4, 10000): 각 프레임을 GRU 입력으로 전달하기 위해 시간순으로 이어붙인다.
    frame3 = keras.layers.Bidirectional(keras.layers.GRU(1250, return_sequences=True))(frame3)
        # (None, 4, 2500): 쌍방향 GRU로 과거와 미래 정보를 반영. 4개의 시퀀스 벡터를 반환.
    frame3 = keras.layers.Reshape((4, 50, 50, 1))(frame3)
    frame3 = keras.layers.concatenate([frame3[:, i] for i in range(4)])
        # (None, 50, 50, 4): 4개의 50*50*1 시퀀스를 겹쳐서 50*50*4 피쳐맵으로 만들었음
    frame3 = keras.layers.Conv2D(4, 3, padding='same', activation='selu')(frame3)
    frame3 = decoder(frame3)
        # (Nome, 100, 100, 3): 최종적인 프레임 형태.
    frame3 = keras.layers.Reshape((1, 100, 100, 3), name='frame3')(frame3)
        # target dataset과 모양을 맞추야 하므로 reshape

    model = keras.Model(inputs=frames, outputs=frame3, name='middle_frame_prediction_model')

    return model
