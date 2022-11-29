"""
오토인코더의 성능을 최대한 끌어 올리기 위해
한층 한층 차근차근 쌓아기며 최적의 오토인코더를 찾아가고자 한다.
"""

import numpy as np
import make_dataset
import os
from tensorflow import keras

import train_networks
from train_networks import Train

_npy_dataset_path = r"D:\data10sec\datasets"
_root_log_path = os.path.join('train', 'logs')
_root_weights_save_path = os.path.join('train', 'checkpoints')


def proto_autoencoder_train_and_save(proto_auto_encoder, num_npy: int = 10, epochs: int = 100, iteration: int = 10,
                                     encoder=None, batch_size: int = 32,
                                     npy_dataset_path=_npy_dataset_path, weights_load_ok=False):
    """
    # 최적 오토인코더 찾는 방법 ( = main코드)
    1. 작은 오토인코더를 훈련하고, 새로운 작은 오토인코더를 훈련하고, 두개를 합쳐서 성능평가
    2. 다시 새로운 작은 오토인코더를 훈련하고, 이전 단계 오토입코더에 합쳐서 성능평가.

    이때 모든 훈련 과정에선 같은 데이터 세트를 사용하도록 해야한다.
    """
    log_path = os.path.join(_root_log_path, proto_auto_encoder.name)
    weights_save_path = os.path.join(_root_weights_save_path, proto_auto_encoder.name + '.h5')
    if weights_load_ok:
        print(weights_save_path)
        proto_auto_encoder.load_weights(weights_save_path)

    # 오토인코더에선 binary cross entropy가 수렴에 더 좋다고 한다. (p679, 핸즈온 머신러닝)
    # mes일땐 loss가 e-7등 매우 작은 숫자였는데 loss 함수 바꾸니까 e-1로 보여준다.
    proto_auto_encoder.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['acc'])
    print(f"Training: {proto_auto_encoder.name} model ")
    for i in range(iteration):
        print(f"{proto_auto_encoder.name}: {i + 1}th iteration starts.")
        print("You can monitor this training by using Tensor Board. Type the next line in terminal window")
        print(f"<tensorboard --logdir={log_path} --port=6006>")
        ndarray, dataset_size = make_dataset.load_npys_and_get_one_np(npy_dataset_path, num_npy=num_npy, num_seq=1,
                                                                      seed=i)  # seed=i: 각 부품 오토인코더를 훈련할 때마다 같은 훈련 세트를 사용하도록 보장.
        trainset, validset, testset = make_dataset.np_to_tf_dataset(ndarray, dataset_size, encoder=encoder, seed=i)

        proto_auto_encoder.fit(trainset, epochs=epochs, batch_size=batch_size, validation_data=validset,
                               callbacks=Train.callbacks(log_path=log_path, weights_save_path=weights_save_path, save_weights_only=False))
        del ndarray, trainset, validset, testset

    proto_auto_encoder.save(os.path.join(os.path.join('train', 'checkpoints'), proto_auto_encoder.name))

    return proto_auto_encoder


''''Residual Unit 적용했으나 성능향상 실패...'''
'''def _RU_for_encoder(tensor, filters, kernel_size_1=3, kernel_size_2=3, strides_1 = 1, strides_2 = 1): # residual unit
    skip_value = keras.layers.Conv2D(filters, 1,  strides=strides_1, padding='same', use_bias=False)(tensor) # 스케일만 맞추고 최대한 입력 형태 그대로 유지.
    skip_value = keras.layers.BatchNormalization()(skip_value)

    ru_output = keras.layers.Conv2D(filters, kernel_size_1, strides=strides_1, padding='same', use_bias=False)(tensor)
    ru_output = keras.layers.BatchNormalization()(ru_output)
    ru_output = keras.layers.Activation('elu')(ru_output)
    ru_output = keras.layers.Conv2D(filters, kernel_size_2, strides=strides_2, padding='same', use_bias=False)(ru_output)
    ru_output = keras.layers.BatchNormalization()(ru_output)
    added_output = keras.layers.Add()([skip_value, ru_output])
    return keras.layers.Activation('elu')(added_output)


def _RU_for_decoder(tensor, filters, kernel_size_1 = 3, kernel_size_2 = 3, strides_1 = 1, strides_2 = 1):
    skip_value = keras.layers.Conv2DTranspose(filters, 1,  strides=strides_1, padding='same', use_bias=False)(tensor) # 스케일만 맞추고 최대한 입력 형태 그대로 유지.
    skip_value = keras.layers.BatchNormalization()(skip_value)

    ru_output = keras.layers.Conv2DTranspose(filters, kernel_size_1, strides=strides_1, padding='same', use_bias=False)(tensor)
    ru_output = keras.layers.BatchNormalization()(ru_output)
    ru_output = keras.layers.Activation('elu')(ru_output)
    ru_output = keras.layers.Conv2DTranspose(filters, kernel_size_2, strides=strides_2, padding='same', use_bias=False)(ru_output)
    ru_output = keras.layers.BatchNormalization()(ru_output)
    added_output = keras.layers.Add()([skip_value, ru_output])
    return keras.layers.Activation('elu')(added_output)'''


def _num1_proto_ae():  # (100, 100, 3) = 30000 -> (50, 50, 9) = 22500 으로 이미지 크기를 압축해주는 오토인코더.
    inputs = keras.layers.Input(shape=(100, 100, 3))  # 1번 오토인코더의 인코더와 디코더는 최종 조합된 오토인코더에서 가장 바깥 부분이 된다.
    encoded = keras.layers.Conv2D(9, 5, strides=2, padding='same', activation='elu')(inputs)
    encoded = keras.layers.Conv2D(9, 3, strides=1, padding='same', activation='elu')(encoded)
    encoder = keras.Model(inputs=inputs, outputs=encoded, name='proto_enc1')

    encoded_inputs = keras.layers.Input(shape=(50, 50, 9))
    decoded = keras.layers.Conv2DTranspose(9, 3, strides=1, padding='same', activation='elu')(encoded_inputs)
    decoded = keras.layers.Conv2DTranspose(9, 5, strides=2, padding='same', activation='elu')(decoded)
    decoded = keras.layers.Conv2D(3, 1)(decoded)
    decoder = keras.Model(inputs=encoded_inputs, outputs=decoded, name='proto_dec1')

    return keras.models.Sequential([encoder, decoder], name='proto_ae1')


def _num2_proto_ae():  # num1_proto_ae 사이에 들어갈 또 하나의 작은 오토 인코더. 훈련시킬땐 num1_proto_ae인코더로 인코딩된 훈련세트로 훈련시켜야 한다.
    inputs = keras.layers.Input(shape=(50, 50, 9))
    encoded = keras.layers.Conv2D(5, 3, strides=1, padding='same', activation='elu')(inputs)
    encoded = keras.layers.Conv2D(5, 3, strides=1, padding='same', activation='elu')(encoded)
    encoder = keras.Model(inputs=inputs, outputs=encoded, name='proto_enc2')

    encoded_inputs = keras.layers.Input(shape=(50, 50, 5))
    decoded = keras.layers.Conv2DTranspose(9, 3, strides=1, padding='same', activation='elu')(encoded_inputs)
    decoded = keras.layers.Conv2DTranspose(9, 3, strides=1, padding='same', activation='elu')(decoded)
    decoder = keras.Model(inputs=encoded_inputs, outputs=decoded, name='proto_dec2')

    return keras.models.Sequential([encoder, decoder], name='proto_ae2')


def _num3_proto_ae():  # (50, 50, 5) -> (50, 50, 3) 으로 압축한다. 1,2,3 proto_ae를 조합해서 테스트한 결과 복원 정확도를 90%~95%를 유지한다.
    inputs = keras.layers.Input(shape=(50, 50, 5))
    encoded = keras.layers.Conv2D(4, 3, strides=1, padding='same', activation='elu')(inputs)
    encoded = keras.layers.Conv2D(4, 3, strides=1, padding='same', activation='elu')(encoded)
    encoder = keras.Model(inputs=inputs, outputs=encoded, name='proto_enc3')

    encoded_inputs = keras.layers.Input(shape=(50, 50, 4))
    decoded = keras.layers.Conv2DTranspose(5, 3, strides=1, padding='same', activation='elu')(encoded_inputs)
    decoded = keras.layers.Conv2DTranspose(5, 3, strides=1, padding='same', activation='elu')(decoded)
    decoder = keras.Model(inputs=encoded_inputs, outputs=decoded, name='proto_dec3')

    return keras.models.Sequential([encoder, decoder], name='proto_ae3')


def _num4_proto_ae():  # 테스트 결과 _num4_proto_ae()까지 포함해서 조합하면 복원 정확도가 80%를 하회한다. 따라서 _num3_proto_ae()까지만 조합하기로 결정.
    inputs = keras.layers.Input(shape=(50, 50, 3))
    encoded = keras.layers.Conv2D(2, 3, strides=1, padding='same', activation='elu')(inputs)
    encoder = keras.Model(inputs=inputs, outputs=encoded, name='proto_enc4')

    encoded_inputs = keras.layers.Input(shape=(50, 50, 2))
    decoded = keras.layers.Conv2DTranspose(3, 3, strides=1, padding='same', activation='elu')(encoded_inputs)
    decoder = keras.Model(inputs=encoded_inputs, outputs=decoded, name='proto_dec4')

    return keras.models.Sequential([encoder, decoder], name='proto_ae4')


if __name__ == '__main__':
    """ gpu 최적화"""
    train_networks.GPU_limit.gpu_limit()
    os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')

    """network._num1_proto_ae 훈련시키기"""
    proto_autoencoder_train_and_save(_num1_proto_ae())

    """network._num2_proto_ae 훈련시키기"""
    # _num1_proto_ae의 인코더와 디코더 사이에 들어갈 또다른 작은 오토인코더. 이 오토인코더를 훈련시키기 위해선
    # _num1_proto_ae의 인코더로 훈련 세트를 변환시켜야 한다.
    proto_ae1_encoder = keras.models.load_model(os.path.join('train', 'checkpoints', 'proto_ae1')).layers[0]
    proto_autoencoder_train_and_save(_num2_proto_ae(), encoder=proto_ae1_encoder)  # encoder인자에 인코더를 넣으면 훈련세트 자동 변환

    """앞선 두개의 오토인코더를 proto_ae_1221 오토인코더로 조합하기"""  # 복원 정확도 95% 내외
    proto_ae1_layers = keras.models.load_model(os.path.join('train', 'checkpoints', 'proto_ae1')).layers
    proto_ae2_layers = keras.models.load_model(os.path.join('train', 'checkpoints', 'proto_ae2')).layers

    proto_ae_1221 = keras.models.Sequential([proto_ae1_layers[0],
                                             proto_ae2_layers[0],
                                             proto_ae2_layers[1],
                                             proto_ae1_layers[1]], name="proto_ae_1221")

    proto_autoencoder_train_and_save(proto_ae_1221)

    """network._num3_proto_ae 훈련시키기"""
    proto_ae_1221 = keras.models.load_model(os.path.join('train', 'checkpoints', 'proto_ae_1221'))
    proto_ae_1221_encoder = keras.models.Sequential([proto_ae_1221.layers[0],
                                                     proto_ae_1221.layers[1]], name="proto_ae_1221_encoder")

    proto_autoencoder_train_and_save(_num3_proto_ae(), encoder=proto_ae_1221_encoder)

    """앞선 세개의 오토인코더를 proto_ae_123321 오토인코더로 조합하기"""
    # 복원 정확도 90% 이상.
    # (100*100*3) -> (50*50*4) 으로 압축하는건 성능이 어느 정도 좋다!  이 이후로 더 층을 늘려봐도 80프로 정도를 하회한다.
    # 따라서 proto_ae_123321를 optimal autoencoder로 결정
    proto_ae1_layers = keras.models.load_model(os.path.join('train', 'checkpoints', 'proto_ae1')).layers
    proto_ae2_layers = keras.models.load_model(os.path.join('train', 'checkpoints', 'proto_ae2')).layers
    proto_ae3_layers = keras.models.load_model(os.path.join('train', 'checkpoints', 'proto_ae3')).layers

    proto_ae_123321 = keras.models.Sequential([proto_ae1_layers[0],
                                               proto_ae2_layers[0],
                                               proto_ae3_layers[0],
                                               proto_ae3_layers[1],
                                               proto_ae2_layers[1],
                                               proto_ae1_layers[1]], name="proto_ae_123321")

    proto_autoencoder_train_and_save(proto_ae_123321)
