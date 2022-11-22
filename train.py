from network import *
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

trainset = np.load(r"D:\smalldata\trainset.npy")

inputs = np.expand_dims(trainset[1], 0)
encoder = keras.layers.Conv2D(16, 3, padding='same', activation='selu')(inputs)  # (1, 100, 100, 16)
encoder = keras.layers.Conv2D(16, 3, padding='same', activation='selu')(inputs)
encoder = keras.layers.AvgPool2D(pool_size=2)(encoder)  # (1, 50, 50, 16)
print(encoder.shape)
encoder = keras.layers.Conv2D(32, 3, padding='same', activation='selu')(encoder)  # (1, 50, 50, 32)
encoder = keras.layers.Conv2D(32, 3, padding='same', activation='selu')(encoder)
encoder = keras.layers.AvgPool2D(pool_size=2)(encoder)  # (1, 25, 25, 32)
print(encoder.shape)
encoder = keras.layers.Conv2D(64, 3, padding='same', activation='selu')(encoder)  # (1, 25, 25, 64)
encoder = keras.layers.Conv2D(64, 3, padding='same', activation='selu')(encoder)
print(encoder.shape)
encoder_model = keras.Model(inputs=inputs, outputs=encoder, name='encoder')

encoder_output = keras.layers.Input(shape=(12, 12, 64))
decoder = keras.layers.Conv2DTranspose(32, 3, strides=2, padding='valid', activation='selu')(encoder_output)
decoder = keras.layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='selu')(decoder)
decoder = keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='selu')(
    decoder)