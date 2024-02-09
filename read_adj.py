import pickle
import time
import numpy as np
import pandas as pd

def generate_train_val_test(stat_file,
                            train_val_test_split=[0.7, 0.1, 0.2]):
    """

    :param stat_file:
    :param masking:
    :param train_val_test_split:
    :param mask_ones_proportion:
    :return: None, save the dataframes into 'npz' files, which are saved under the same path of 'stat_file'
        x_train/val/test: (N, L, D)
        dateTime: (N, L)
        y_train_val_test: (N, L, M)
    """
    start = time.time()
    stat_data = np.load(stat_file,allow_pickle=True)
    x = stat_data['speed_sequences']
    dateTime = stat_data['dateTime']
    y = stat_data['speed_labels']
    max_speed = stat_data['max_speed']

    print("x shape: ", x.shape, "dateTime shape: ", dateTime.shape, ", y shape: ", y.shape)

    # Write the data into npz file.
    num_samples = x.shape[0]
    num_train = round(num_samples * train_val_test_split[0])
    num_test = round(num_samples * train_val_test_split[2])
    num_val = num_samples - num_test - num_train

    x_train, dateTime_train, y_train = x[:num_train], dateTime[:num_train], y[:num_train]
    x_val, dateTime_val, y_val = (
        x[num_train: num_train + num_val],
        dateTime[num_train: num_train + num_val],
        y[num_train: num_train + num_val]
    )
    x_test, dateTime_test, y_test = x[-num_test:], dateTime[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        # locals() 函数会以字典类型返回当前位置的全部局部变量
        _x, _dateTime, _y= locals()["x_" + cat], locals()["dateTime_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        # x: (N, L, D)
        # dateTime: (N, L)
        # y: (N, L, M)

        file_save_path = stat_file[:-4] + '_' + cat + '.npz'  # e.g., 'x.npz' -> 'x_train.npz'
        # np.savez_compressed(
        #     file=file_save_path,
        #     x=_x,
        #     dateTime=_dateTime,
        #     y=_y,
        #     max_speed=max_speed
        # )
    print("The data splitting is finised in {}s with splitting ratio: {}".format(time.time() - start,
                                                                                 str(train_val_test_split)))
    return

def generate_dataset(traffic_df_filename):
    """
            To generate the statistic features from raw datasets and save them into "npz" files
        :param traffic_df_filename:
        :param output_dir: the path to save generated datasets
        :return:
            df: (N_all, D), the full dataframe including "dateTime" ass the first column
            save datasets into ".npz" files
            # x: (N, L, D)
            # dateTime: (N, L)
            # y: (N, L, M) M indicates the number of missing nodes
        """

    # load data
    df = pd.read_hdf(traffic_df_filename)
    print(df.shape)
    # Predict the same time period
    x_offsets = np.sort(
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (N, L, D)
    # y: (N, L, M)

    num_samples, num_nodes = df.shape
    data = df.values  # (num_samples, num_nodes)
    # 将值范围外的值进行归缩
    speed_tensor = data.clip(0, 100)  # (N, D)
    max_speed = speed_tensor.max().max()
    # 为了让数字不失去相对意义,我们需要进行量纲化 (最大值化)
    speed_tensor = speed_tensor / max_speed  # (N, D)

    # 同时找到日期索引
    date_array = df.index.values  # (N)
    # print(speed_tensor.shape, date_array.shape)

    x, dateTime, y = [], [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = speed_tensor[t + x_offsets, ...]
        dateTime_t = date_array[t + x_offsets]
        y_t = speed_tensor[t + y_offsets, ...]
        x.append(x_t)
        dateTime.append(dateTime_t)
        y.append(y_t)

    # 按顺序堆叠数组
    speed_sequences = np.stack(x, axis=0)  # (N, L, D)
    dateTime = np.stack(dateTime, axis=0)  # (N, L)
    speed_labels = np.stack(y, axis=0)  # (N, L, M)


'''
pickle_file = 'Datasets/TJEC/gis_adj.pkl'

with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f, encoding='latin1')

# pickle_data = pickle_data[2]
pos = pickle_data[pickle_data > 0]
count = len(pos)
# count = len(pos) - len(pickle_data)
print(pickle_data)
'''

stat_file = 'Datasets/TJEC/tjec.h5'
stat_file_2 = 'Datasets/TJEC/Time_sequence.npz'

train_val_test_split = [0.7, 0.1, 0.2]
generate_dataset(stat_file)
generate_train_val_test(stat_file_2, train_val_test_split)
