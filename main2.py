from network import *
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

auto_import()
gpu_limit()

trainset = np.load(r"D:\smalldata\trainset.npy")
trainset.shape

trainset.shape[0]//3
trainset2 = trainset[:trainset.shape[0]//3*3].reshape(trainset.shape[0]//3,3,100,100,3)
plt.imshow(trainset2[0,2])


'''trainset = np.load(r"D:\smalldata\trainset.npy")

validset = np.load(r"D:\smalldata\validset.npy")

encoder, decoder, autoencoder = autoencoder()
autoencoder.compile(loss='mse', optimizer='adam', metrics=['acc'])

checkpoint_cb = keras.callbacks.ModelCheckpoint(r"D:\smalldata\autoencoder_checkpoint.h5", save_best_only = True) # 마지막 인자는 fit할때 validation 세트가 있어야 설정 가능.
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)  # 10에포크 동안 검증 세트에 대한 성능향상이 없을 경우 훈련 조기 종료. 최상 모델로 저장됨(history 객체로)
history = autoencoder.fit(x=trainset[:3000], y=trainset[:3000], epochs = 1000, validation_data=(trainset[-500:], trainset[-500:]), callbacks =  [checkpoint_cb, early_stopping_cb])

model = keras.models.load_model(r"D:\smalldata\autoencoder_checkpoint.h5")
pred = model.predict(trainset[:3])
for i in range(3):
    plt.imshow(pred[i])
plt.imshow(pred[1])
plt.imshow(trainset[i])'''

#history = model.fit(x=trainset[6000:9000], y=trainset[6000:9000], epochs = 1000, validation_data=(trainset[-1500:-1000], trainset[-1500:-1000]), callbacks =  [checkpoint_cb, early_stopping_cb])




'''trainset = npys_to_one_np("D:\smalldata\datasets")
validset = npys_to_one_np(r"D:\valdata\datasets")
np.random.shuffle(trainset)
np.random.shuffle(validset)
np.save(r'D:\smalldata\trainset', trainset)
np.save(r'D:\smalldata\validset', validset)'''




