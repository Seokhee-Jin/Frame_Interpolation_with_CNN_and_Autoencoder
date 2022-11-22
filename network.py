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

def npys_to_one_np(npysDirPath: str):
    npyPathList = [os.path.join(npysDirPath, filename) for filename in os.listdir(npysDirPath) if
                   filename.endswith('.npy')]

    movies = np.load(npyPathList[0])
    print(f'{npyPathList[0]} is appended to return. --{1}/{len(npyPathList)}--')
    for i in range(len(npyPathList) - 1):
        movies = np.append(movies, np.load(npyPathList[i + 1]), axis=0)
        print(f'{npyPathList[i+1]} is appended to return.--{(i+2)}/{len(npyPathList)}--')

    movies = (movies/255.0).astype('float32')

    return  movies


def autoencoder():
    inputs = keras.layers.Input(shape=(100, 100, 3))  # (1, 100, 100, 3)
    encoder = keras.layers.Conv2D(16, 3, padding='same', activation='selu')(inputs)  # (1, 100, 100, 16)
    encoder = keras.layers.Conv2D(16, 3, padding='same', activation='selu')(inputs)
    encoder = keras.layers.AvgPool2D(pool_size=2)(encoder)  # (1, 50, 50, 16)
    encoder = keras.layers.Conv2D(32, 3, padding='same', activation='selu')(encoder)  # (1, 50, 50, 32)
    encoder = keras.layers.Conv2D(32, 3, padding='same', activation='selu')(encoder)
    encoder = keras.layers.AvgPool2D(pool_size=2)(encoder)  # (1, 25, 25, 32)
    encoder = keras.layers.Conv2D(64, 3, padding='same', activation='selu')(encoder)  # (1, 25, 25, 64)
    encoder = keras.layers.Conv2D(64, 3, padding='same', activation='selu')(encoder)
    encoder_model = keras.Model(inputs=inputs, outputs=encoder, name='encoder')

    encoder_output = keras.layers.Input(shape=(25, 125, 64))
    decoder = keras.layers.Conv2DTranspose(32, 3, strides=2, padding='valid', activation='selu')(encoder_output)
    decoder = keras.layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='selu')(decoder)
    decoder = keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='selu')(
        decoder)  # (1, 100, 100, 3)
    decoder_model = keras.Model(inputs=encoder_output, outputs=decoder, name='decoder')

    autoencoder_model = keras.models.Sequential([encoder_model, decoder_model])

    return encoder_model, decoder_model, autoencoder_model

''''''


if __name__ == '__main__':

    gpu_limit2()
    trainset = npys_to_one_np("D:\smalldata\datasets")
    validset = npys_to_one_np(r"D:\valdata\datasets")


    trainset = np.random.shuffle(validset)
    validset = np.random.shuffle(trainset)




    encoder, decoder, autoencoder = autoencoder()
    autoencoder.compile(loss='mse', optimizer='adam')
    history = autoencoder.fit(x=trainset, y=trainset, batch_size=4)

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
