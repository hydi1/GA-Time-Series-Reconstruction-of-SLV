import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


class Dataset_Npy(Dataset):
    def __init__(self, root_path, data_path, target_path, flag='train', size=None,
                 features='S', target='OT', scale=False, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.target_path = target_path
        self.__read_data__()
        self.scaler_x = None # 用于输入数据 x 的 scaler
        self.scaler_y =None # 用于目标数据 y 的 scaler
        self.__read_data__()

    def __read_data__(self):

        data_raw = np.load(os.path.join(self.root_path, self.data_path))
        self.data_x = data_raw.reshape(-1, 4*29*6*11) # 输入特征
        # print(f"重塑后的数据形状: {self.data_x.shape}")
        df_target = pd.read_excel(self.target_path)
        self.data_y = df_target.iloc[:,-1].values  # 目标数据
        print("data_y.shape=", self.data_y.shape)#(546,1)

        self.time = df_target.iloc[:, 0].values

        num_train = int(len(self.data_x) * 0.6)
        print("num_train=", num_train)#220
        num_test = int(len(self.data_x) * 0.1)
        print("num_test=", num_test)#36
        num_valid = int(len(self.data_x) * 0.3)
        print("num_valid=", num_valid)#110
        #数据集是按时间顺序切分的
        border1s = [0, num_train - self.seq_len, len(self.data_x) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(self.data_x)]
        border1 = border1s[self.set_type]
        # print("border1=", border1s[0], border1s[1], border1s[2])#border1= 0 450 529
        border2 = border2s[self.set_type]
        # print("border2=", border2s[0], border2s[1], border2s[2])#border2= 546 624 781
        #训练数据范围

        flag_x = self.data_x[border1:border2]
        # print("data_x.shape=", self.data_x.shape)#(546,441)
        flag_y = self.data_y[border1:border2]
        # print("data_y.shape=", self.data_y.shape)#(546,1)

        # 归一化输入数据 x
        if self.scale:  # 假设 self.scale 是一个布尔变量，控制是否进行归一化
            train_data_x = data_raw.reshape(-1,4*29*6*11)[0:num_train]
            self.scaler_x = StandardScaler()
            # 这是固定的训练集的归一化
            train_data_x_fit = self.scaler_x.fit_transform(train_data_x)
            joblib.dump(self.scaler_x, 'scaler_x_time.pkl')
            if self.set_type == 0:  # 训练集
                self.data_x = train_data_x_fit
            elif self.set_type == 1:  # 验证集
                self.data_x = self.scaler_x.transform(flag_x)
            else:  # 测试集
                self.data_x = self.scaler_x.transform(flag_x)
        else:
            if self.set_type == 0:  # 训练集
                self.data_x = data_raw.reshape(-1,4*29*6*11)[0:num_train]
            elif self.set_type == 1:  # 验证集
                self.data_x = flag_x
            else:  # 测试集
                self.data_x = flag_x

        # 归一化目标数据 y
        if self.scale:  # 假设 self.scale 是一个布尔变量，控制是否进行归一化
            train_data_y = self.data_y[0:num_train]
            # print("train_data_y.shape=", train_data_y.reshape(-1, 1).shape)
            self.scaler_y = StandardScaler()
            train_data_y_fit = self.scaler_y.fit_transform(train_data_y.reshape(-1, 1))
            joblib.dump(self.scaler_y, 'scaler_y_time.pkl')
            if self.set_type == 0:  # 训练集
                self.data_y = train_data_y_fit
            elif self.set_type == 1:  # 验证集
                self.data_y = self.scaler_y.transform(flag_y.reshape(-1, 1))
                print("flag_y shape before transform:", flag_y.shape)

            else:  # 测试集

                print("flag_y shape before transform:", flag_y.shape)
                self.data_y = self.scaler_y.transform(flag_y.reshape(-1, 1))
        else:
            if self.set_type == 0:  # 训练集
                self.data_y = df_target.values[0:num_train]
            elif self.set_type == 1:  # 验证集
                self.data_y = flag_y
            else:  # 测试集
                self.data_y = flag_y
        # print('self.data_y.shape',self.data_y.shape)

        df_stamp = pd.to_datetime(self.time)

        # 将 'date' 列转换为 DatetimeIndex
        dates = pd.DatetimeIndex(df_stamp)

        # 提取日期的时间特征
        self.data_stamp = time_features(dates, freq='D')
        self.data_stamp = self.data_stamp.transpose(1, 0)


    def __getitem__(self, index):
        seq_x = self.data_x[index:index + self.seq_len]  # 输入序列
        # print("seq_x.shape=", seq_x.shape)#12, 461760
        seq_y = self.data_y[index:index + self.seq_len]  # 目标序列，长度与 seq_x 相同
        # print("seq_y.shape=", seq_y.shape)#(seq_len,1) (12, 1)
        # 如果有时间标记需求，用实际数据替代全零张量
        seq_x_mark = self.data_stamp[index:index + self.seq_len]
        # print("seq_x_mark.shape=", seq_x_mark.shape)# (12, 3)
        seq_y_mark = self.data_stamp[index:index + self.seq_len] # 根据实际情况修改


        # 确保 seq_y 为二维 [seq_len, 1]
        if len(seq_y.shape) == 1:
            seq_y = seq_y[:, np.newaxis]

        return (
            # torch.from_numpy(seq_x).float(),
            # torch.from_numpy(seq_y).float(),
            seq_x,
            seq_y,
            seq_x_mark,
            seq_y_mark,
        )

    def __len__(self):
        # 它定义了数据集的长度，有多少个批次或样本可以用于训练，因为构建数据集时采用的是滑动窗口的方式
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data, is_target=True):
        if is_target and self.scaler_y is None:
            raise ValueError("Scaler for target data has not been initialized.")
        if not is_target and self.scaler_x is None:
            raise ValueError("Scaler for input data has not been initialized.")
        if is_target:
            return self.scaler_y.inverse_transform(data)
        else:
            return self.scaler_x.inverse_transform(data)









