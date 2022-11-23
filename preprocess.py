'''
num_frames*100*100*3 포맷의 비디오 npy를 여러개 로드해서 하나의 tf dataset으로 반환하기 위한 함수들을 정의.
'''
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

def load_npys_and_get_one_np(folderpath: str, num_npy: int, num_seq: int = 3, batch: int = 32, seed: int = None):
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
    return (np_of_npys / 255.0).astype('float32')


def np_to_tf_dataset(ndarray, buffer_size:int = 10000, batch_size: int = 32, seed: int = None):
    dataset = tf.data.Dataset.from_tensor_slices(ndarray)
    dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size=batch_size, drop_remainder=True).prefetch(1)
    return dataset

if __name__ == '__main__':
    dataset = np_to_tf_dataset(load_npys_and_get_one_np(r"D:\data1min_\datasets", 3))
    for item in dataset.take(1):
        print(item.shape)
        for i in range(3):
            for j in range(3):
                plt.imshow(item[i, j])
                plt.show()
