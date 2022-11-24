import network
from network import *
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from preprocess import *
from network import *

np_, size_ = load_npys_and_get_one_np(r"D:\data1min_\datasets", 3, num_seq=1, seed=42)
print(10, size_)
trainset, validset, testset = np_to_tf_dataset(np_,size_)

print(12)
encoder, decoder, autoencoder = autoencoder()

'''for item in trainset.take(3):
    print(type(item[0]))
    print(item[0].shape)
'''

autoencoder.compile(loss='mse', optimizer='adam',metrics=['acc'])

checkpoint_cb = keras.callbacks.ModelCheckpoint(r".\autoencoder_checkpoint.h5", save_best_only = True) # 마지막 인자는 fit할때 validation 세트가 있어야 설정 가능.
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)  # 10에포크 동안 검증 세트에 대한 성능향상이 없을 경우 훈련 조기 종료. 최상 모델로 저장됨(history 객체로)

history = autoencoder.fit(trainset, epochs=100, batch_size=32,  validation_data=validset, callbacks =  [checkpoint_cb, early_stopping_cb])
