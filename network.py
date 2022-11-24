#pip install --user tensorflow_gpu==2.11.0  # 그래픽카드 적용

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
def auto_import():
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import cv2
    keras.layers.Reshape()

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
        encoder = keras.layers.Conv2D(12, 3, strides=2, padding='same', activation='selu')(inputs)
        encoder = keras.layers.Conv2D(12, 3, padding='same', activation='selu')(inputs)
        encoder = keras.layers.Conv2D(24, 3, strides=2, padding='same', activation='selu')(encoder)
        encoder = keras.layers.Conv2D(24, 3, padding='same', activation='selu')(encoder)
        encoder = keras.layers.Conv2D(48, 3, strides=2, padding='same', activation='selu')(encoder)
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
        decoder = keras.layers.Conv2DTranspose(24, 3, strides=2, padding='same', activation='selu')(decoder)
        decoder = keras.layers.Conv2DTranspose(12, 3, strides=2, padding='same', activation='selu')(decoder)
        decoder = keras.layers.Conv2DTranspose(3, 3, strides=1, padding='same', activation='selu')(
            decoder)
        decoder_model = keras.Model(inputs=encoder_output, outputs=decoder, name='decoder')
        return decoder_model

    autoencoder_model = keras.models.Sequential([encoder(), decoder()])

    return encoder(), decoder(), autoencoder_model

''''''


if __name__ == '__main__':


    encoder, decoder, autoencoder = autoencoder()
    autoencoder.compile(loss='mse', optimizer='adam')


'''movies = movie1
for movie in
print(np.append(movie1, movie1, axis=0).shape)


np_big = np.load("D:\data\datasets\Avatar.npy")
np.big[...,1]
tf_big = tf.data.Dataset(np_big)

np_0 = np.load(r"D:\smalldata\datasets\Avatar.npy")
tf_0 = tf.data.Dataset.from_tensor_slices(np_0)
tf_1 = tf_0.shuffle(10000, seed=42).batch(1440, drop_remainder=True) # 순차성을 없애기 위해 10000으로 섞고, 편의를 위해 1분 단위(24*60=1440)로 묶음
count = 0
for item in tf_1:
    count += 1
print(count)
'''



"""
'''npypaths making'''
datasets_dir = r"D:\data\datasets"
npylist = os.listdir(datasets_dir)
npypaths = list(map(lambda x: os.path.join(datasets_dir, x), npylist))#완전한 경로로 만들기
npypaths

#filepath_dataset = tf.data.Dataset.list_files(filepaths, seed=42) # 파일경로 리스트를 섞어서 데이터셋 타입으로 반환.
''''''


'''shard making'''
num = 0
for npypath in npypaths:
    np_ = np.load(npypath)
    tf_ = tf.data.Dataset.from_tensor_slices(np_)
    tf_.save(os.path.join(r"D:\data\shards", f'shard{num}'))
    num+=1


shards_dir = r"D:\data\shards"
shardlist = os.listdir(shards_dir)
shardpaths = list(map(lambda x: os.path.join(shards_dir, x), shardlist))
shardpaths

"""
