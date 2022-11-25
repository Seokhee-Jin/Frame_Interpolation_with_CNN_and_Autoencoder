#pip install --user tensorflow_gpu==2.11.0  # 그래픽카드 적용

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


def autoencoder():
    def encoder():
        inputs = keras.layers.Input(shape=(100, 100, 3))
        encoder = keras.layers.Dropout(rate=0.2)(inputs)
        encoder = keras.layers.Conv2D(12, 3, strides=2, padding='same', activation='selu')(inputs)
        encoder = keras.layers.Dropout(rate=0.2)(encoder)
        encoder = keras.layers.Conv2D(12, 3, padding='same', activation='selu')(inputs)
        encoder = keras.layers.Dropout(rate=0.2)(encoder)
        encoder = keras.layers.Conv2D(24, 3, strides=2, padding='same', activation='selu')(encoder)
        encoder = keras.layers.Dropout(rate=0.2)(encoder)
        encoder = keras.layers.Conv2D(24, 3, padding='same', activation='selu')(encoder)
        encoder = keras.layers.Dropout(rate=0.2)(encoder)
        encoder = keras.layers.Conv2D(48, 3, strides=2, padding='same', activation='selu')(encoder)
        encoder = keras.layers.Dropout(rate=0.2)(encoder)
        encoder = keras.layers.Conv2D(48, 3, padding='same', activation='selu')(encoder)

        encoder = keras.layers.Flatten()(encoder)
        encoder = keras.layers.Dropout(rate=0.2)(encoder)
        encoder = keras.layers.Dense(2500, activation='elu', kernel_initializer='he_normal')(encoder)
        encoder = keras.layers.Reshape((50, 50))(encoder)
        encoder_model = keras.Model(inputs=inputs, outputs=encoder, name='encoder')
        return encoder_model

    def decoder():
        encoder_output = keras.layers.Input(shape=(50, 50))
        decoder = keras.layers.Flatten()(encoder_output)
        decoder = keras.layers.Dropout(rate=0.2)(decoder)
        decoder = keras.layers.Dense(30000, activation='elu', kernel_initializer='he_normal')(decoder)
        decoder = keras.layers.Reshape((25, 25, 48))(decoder)
        decoder = keras.layers.Dropout(rate=0.2)(decoder)
        decoder = keras.layers.Conv2DTranspose(24, 3, strides=2, padding='same', activation='selu')(decoder)
        decoder = keras.layers.Dropout(rate=0.2)(decoder)
        decoder = keras.layers.Conv2DTranspose(12, 3, strides=2, padding='same', activation='selu')(decoder)
        decoder = keras.layers.Dropout(rate=0.2)(decoder)
        decoder = keras.layers.Conv2DTranspose(3, 3, strides=1, padding='same', activation='selu')(
            decoder)
        decoder_model = keras.Model(inputs=encoder_output, outputs=decoder, name='decoder')
        return decoder_model

    autoencoder_model = keras.models.Sequential([encoder(), decoder()])

    return autoencoder_model

def middle_frame_prediction_model(num_seq: int, ae_load_path: str = r".\train\checkpoints\autoencoder_checkpoint.h5"):
    '''

    :param ae_load_path: 오토 인코더를 로드한다. 인코더를 이용해 입력데이터의 차원을 줄인다.
    :return: 예측된 가운데 프레임이 반환된다.
    '''
    auto_encoder = autoencoder()
    auto_encoder.load_weights(ae_load_path)
    encoder = auto_encoder.layers[0]
    encoder.trainable = False # 오토 인코더는 이미 학습을 마쳤다고 가정. 따라서 가중치 갱신 중지

    frames = keras.layers.Input(shape=(4, 100, 100, 3)) #네개 프레임에 각각 인코더를 적용해 차원을 축소한다.
    frame1 = keras.layers.Reshape((100, 100, 3), name='frame1')(frames[:, 0]) #프레임1
    frame1 = encoder(frame1)
    frame1 = keras.layers.Reshape((1, 2500))(frame1)

    frame2 = keras.layers.Reshape((100, 100, 3), name='frame2')(frames[:, 1]) #프레임2
    frame2 = encoder(frame2)
    frame2 = keras.layers.Reshape((1, 2500))(frame2)

    frame4 = keras.layers.Reshape((100, 100, 3), name='frame4')(frames[:, 2]) #프레임4
    frame4 = encoder(frame4)
    frame4 = keras.layers.Reshape((1, 2500))(frame4)

    frame5 = keras.layers.Reshape((100, 100, 3), name='frame5')(frames[:, 3]) #프레임5
    frame5 = encoder(frame5)
    frame5 = keras.layers.Reshape((1, 2500))(frame5)

    frame3 = keras.layers.concatenate([frame1, frame2, frame4, frame5], axis=1) # 각 프레임을 RNN을 위해 시간순으로 이어붙인다.
    frame3 = keras.layers.Bidirectional(keras.layers.GRU(3750, return_sequences=True))(frame3) # 쌍방향 GRU로 과거와 미래 정보를 반영.
    frame3 = keras.layers.Reshape((1, 100, 100, 3), name='frame3')(frame3) # 최종적으로 예측되는 프레임3

    model = keras.Model(inputs=frames, outputs=frame3)

    return model

