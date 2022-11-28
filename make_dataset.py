
"""
 num_frames*100*100*3 포맷의 비디오 npy를 랜덤으로 여러개 로드해서 하나의 tf dataset으로 반환하기 위한 함수들을 정의.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

_npy_dataset_path = r"D:\data1min_\datasets"

def load_npys_and_get_one_np(folderpath: str, num_npy: int, num_seq: int = 3, batch: int = 32, seed: int = None):
    """
    npy 파일들을 로드해서 (ndarray, ndarray.shape[0]) 반환하는 함수이다. \n
    :param folderpath: npy파일들이 있는 폴더 경로. 각각이 (num_frames, 100, 100, 3) 넘파이 배열이다.
    :param num_npy: 랜덤으로 몇개의 npy를 추출할 것인지
    :param num_seq: num_seq>1일 경우 배열 reshape. -> ("number of frames"//num_seq, num_seq, 100, 100, 3).
    :param batch: batch size
    :param seed: npy 샘플링 시드 & 셔플 시드
    :return: (np_of_npys:numpy.ndarray, dataset_size:int)
    """
    np.random.seed(seed)
    npy_names_in_dir = os.listdir(folderpath)
    npy_names = np.random.choice(npy_names_in_dir, num_npy, replace=False)  # npy를 랜덤하게 로드.셔플1
    npy_paths = [os.path.join(folderpath, name) for name in npy_names]

    np_of_npys = np.empty((0, num_seq, 100, 100, 3)) if num_seq > 1 else np.empty((0, 100, 100, 3))
    for npy_path in npy_paths:
        np_of_npy = np.load(npy_path)
        if num_seq > 1:  # 연속된 프레임들을 한개의 데이터로 다룰 경우 reshape.
            num_frame = np_of_npy.shape[0]
            valid_total_num_frame = num_frame - (num_frame % num_seq)
            np_of_npy = np_of_npy[:valid_total_num_frame].reshape(int(valid_total_num_frame / num_seq),
                                                                  num_seq, 100, 100, 3)
            np.random.shuffle(np_of_npy)  # 셔플2
        np_of_npys = np.append(np_of_npys, np_of_npy, axis=0)
    np.random.shuffle(np_of_npys)  # 셔플3
    np_of_npys = (np_of_npys / 255.0).astype('float32') # 수렴이 되지 않은 것 같아서 32 -> 64로 늘렸다./늘리고 싶었지만 메모리 부족으로 다시 32로..
    dataset_size = np_of_npys.shape[0]
    return np_of_npys, dataset_size


def np_to_tf_dataset(ndarray, dataset_size: int, encoder=None, train_split=0.7, val_split=0.2, test_split= 0.1,
                     buffer_size: int = 10000, batch_size: int = 32, seed: int = None):
    """
    load_npys_and_get_one_np 에서 반환받은 ndarray와 dataset_size를 전달하면 훈련 세트와 검증 세트로 구성된
    하나의 tf.data.dataset으로 반환해주는 함수이다. \n
    """
    assert (train_split + test_split + val_split) == 1

    if ndarray.ndim == 5: # middle_frame_predictor용 데이터셋 (x.shape = (None, 2, 100, 100, 3), y.shape = (None, 1, 100, 100, 3))
        center_idx = ndarray.shape[1]//2
        mask_x = [True] * ndarray.shape[1]; mask_x[center_idx] = False
        mask_y = [False] * ndarray.shape[1]; mask_y[center_idx] = True
        # 가운데 프레임이 타겟 데이터가 되도록 한다.
        dataset = tf.data.Dataset.from_tensor_slices((ndarray[:, mask_x], ndarray[:, mask_y]))
    elif encoder is not None: # 인코딩된 데이터셋이 필요할 경우 데이터셋. FindOptimalAutoencoder 에서 사용. (x = encoder(x), y = encoder(x))
        dataset = tf.data.Dataset.from_tensor_slices((encoder(ndarray), encoder(ndarray)))
    else: # autoencoder용 데이터셋 (x.shape = (None, 100, 100, 3), y.shape = (None, 100, 100, 3))
        # autoencoder는 특성값과 타겟값이 같다.
        dataset = tf.data.Dataset.from_tensor_slices((ndarray, ndarray))


    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    train_ds = dataset.take(train_size).shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size=batch_size, drop_remainder=True).prefetch(1)
    val_ds = dataset.skip(train_size).take(val_size).shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size=batch_size, drop_remainder=True).prefetch(1)
    test_ds = dataset.skip(train_size).skip(val_size).shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size=batch_size, drop_remainder=True).prefetch(1)

    return train_ds, val_ds, test_ds

if __name__ == '__main__':
    dataset, _ = np_to_tf_dataset(load_npys_and_get_one_np(_npy_dataset_path, 3))
    for item in dataset.take(1):
        print(item.shape)
        for i in range(3):
            for j in range(3):
                plt.imshow(item[i, j])
                plt.show()
