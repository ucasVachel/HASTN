import os, math
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import geopy.distance
from concurrent.futures import ProcessPoolExecutor

'''
    Input:  "xxx.csv" with first column as "Date"
    Output: the saved files for preprocessed datasets, i.e., "train/val/test.npz" including:
        - x
        - y
        - x_offsets
'''


def generate_dataset(traffic_df_filename, output_dir):
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

    np.savez_compressed(
        output_dir + "Time_sequence.npz",
        speed_sequences=speed_sequences,
        dateTime=dateTime,
        speed_labels=speed_labels,
        max_speed=max_speed
    )


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
    stat_data = np.load(stat_file)
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
        np.savez_compressed(
            file=file_save_path,
            x=_x,
            dateTime=_dateTime,
            y=_y,
            max_speed=max_speed
        )
    print("The data splitting is finised in {}s with splitting ratio: {}".format(time.time() - start,
                                                                                 str(train_val_test_split)))
    return

def retrieve_hist(dateTime, full_data, nh, nd, nw, tau):
    # 得到多尺度观测信息，分为小时、每天、每周
    #
    """

    :param dateTime: (B, L), numpy array
    :param full_data: (N, D) dataframe, with "dateTime" as the first column
    :return:
        A concatenated segment
    """

    B, L = dateTime.shape
    offsets = np.sort(np.arange(0, L, 1))
    full_data_index = full_data.index
    full_data_value = full_data.values
    Td = 12 * 24  # 12 measures per hour
    Tw = 12 * 24 * 7  # 12 measures per hour

    res_h, res_d, res_w = [], [], []
    if tau is None:
        tau = L

    for i in range(B):
        start_date, end_date = dateTime[i, 0], dateTime[i, L - 1]
        start, end = full_data_index.get_loc(start_date), full_data_index.get_loc(end_date)

        # recent observations
        start_h, end_h = start - nh * tau, end - L
        if start_h < 0:  # fill with current observation when no previous readings
            x = np.tile(full_data_value[start:end + 1], (nh * tau, 1, 1))  # (L, D) -> (nh*tau, L, D)
        else:
            x = []
            for t in range(start_h, end_h + 1):  # [start_h, end_h]
                x_t = full_data_value[t + offsets]  # (L, D)
                x.append(x_t)
            x = np.stack(x, axis=0)  # (nh*tau, L, D)
        res_h.append(x)

        # daily observations
        x_d = []
        curr_reading = np.tile(full_data_value[start:end + 1], (tau, 1, 1))  # (tau, L, D)
        for i in range(1, nd + 1, 1):
            start_d, end_d = start - i * Td - int(tau / 2), end - i * Td - L + int(tau / 2)
            if start_d < 0:
                x_d.append(curr_reading)
            else:
                x = []
                for t in range(start_d, end_d + 1):  # [start_d, end_d]
                    x_t = full_data_value[t + offsets]  # (L, D)
                    x.append(x_t)
                x = np.stack(x, axis=0)  # (tau, L, D)
                curr_reading = x
                x_d.append(x)
        x_d = np.concatenate(x_d, axis=0)  # (nd*tau, L, D)
        res_d.append(x_d)

        # weekly observations
        x_w = []
        curr_reading = np.tile(full_data_value[start:end + 1], (tau, 1, 1))  # (tau, L, D)
        for i in range(1, nw + 1, 1):
            start_w, end_w = start - i * Tw - int(tau / 2), end - i * Tw - L + int(tau / 2)
            if start_w < 0:
                x_w.append(curr_reading)
            else:
                x = []
                for t in range(start_w, end_w + 1):  # [start_d, end_d]
                    x_t = full_data_value[t + offsets]  # (L, D)
                    x.append(x_t)
                x = np.stack(x, axis=0)  # (tau, L, D)
                curr_reading = x
                x_w.append(x)
        x_w = np.concatenate(x_w, axis=0)  # (nw*tau, L, D)
        res_w.append(x_w)

    res_h = np.stack(res_h, axis=0)  # (B, nh*tau, L, D)
    res_d = np.stack(res_d, axis=0)  # (B, nd*tau, L, D)
    res_w = np.stack(res_w, axis=0)  # (B, nw*tau, L, D)

    return np.concatenate((res_h, res_d, res_w), axis=1)  # (B, nw*tau + nd*tau + nh *tau, L, D)

if __name__ == "__main__":
    root_path = "../Datasets/"
    datasets = ["PEMS-BAY/", "METR-LA/","TJEC/"]
    dataset = datasets[2]
    data_path = root_path + dataset  # "PEMS-BAY"

    traffic_df_filename = data_path + dataset[:-1].lower() + '.h5'  # raw_hdf file
    # dist_filename = data_path + "graph_sensor_locations.csv"
    '''
    read gis
    '''

    output_dir = data_path
    L = 12
    print("Generate dataset...")
    generate_dataset(traffic_df_filename, output_dir)
    print("Generate dataset is finiesed!")

    generate_train_val_test(output_dir+'Time_sequence.npz')
