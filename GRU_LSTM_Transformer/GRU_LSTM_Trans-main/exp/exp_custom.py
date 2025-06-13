import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from exp.exp_basic import Exp_Basic
from utils.timefeatures import time_features
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, data, seq_len, pred_len, freq='h', mode='pred', stride=1):
        self.data = data
        #输入序列的长度（时间步数）
        self.seq_len = seq_len
        self.pred_len = pred_len
        # 时间序列的频率，如 'h' 表示小时，'d' 表示天。
        self.freq = freq
        self.mode = mode
        self.stride = stride
        self.__read_data__()

    def __read_data__(self):
        """
        self.data.columns: ['date', ...(other features)]
        目的是 读取并处理输入数据，将其分为 特征数据self.data_x和时间特征 self.data_stamp
        """
        if 'date' in self.data.columns:
            #如果有 date列 则提取除 date列 以外的所有列作为特征数据
            cols_data = self.data.columns[1:]
            #将 date 列之外的所有列的数据转换为 NumPy 数组，存储到 self.data_x 中，作为特征数据
            self.data_x = self.data[cols_data].values
            #提取 date 列并存储为一个 DataFrame，赋值给 df_stamp。
            #使用 Pandas 的 pd.to_datetime 方法将 date 列转换为时间格式。
            #调用 time_features 函数，基于日期生成时间特征，例如月份、星期几等（time_features 是自定义函数，可能基于 freq 参数生成不同的时间特征）。
            #self.freq 是该类的另一个属性，用于指定时间特征的频率，例如每日、每小时等
            df_stamp = self.data[['date']]
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            #time_features 返回一个矩阵，默认是形状为 [特征数量, 时间点数量]。
            #转置后，变成 [时间点数量, 特征数量]，赋值给 self.data_stamp，用于后续处理
            self.data_stamp = data_stamp.transpose(1, 0)
        else:
            #没有date列 ，直接使用数据的所有值  
            self.data_x = self.data.values
            #设置 self.data_stamp 为零矩阵，表示无时间戳特征
            self.data_stamp = np.zeros((self.x.shape[0], 1))

    def __getitem__(self, index):
        #获取单个样本，根据index提取时间序列的一个片段
        if self.mode != 'pred':#非预测模式
            #s_begin 是输入序列的起始位置，基于索引 index 和步长 stride 计算
            s_begin = index * self.stride
            #s_end 是输入序列的结束位置，长度为 seq_len
            s_end = s_begin + self.seq_len
            #r_begin 是目标序列的起始位置，紧接着输入序列结束
            r_begin = s_end
            #r_end 是目标序列的结束位置，长度为 pred_len
            r_end = r_begin + self.pred_len
            #提取序列数据 seq_x:从data_x中提取输入序列
            seq_x = self.data_x[s_begin:s_end]
            #提取序列数据 seq_y:从data_x中提取目标序列
            seq_y = self.data_x[r_begin:r_end]
            #提取时间特征 seq_x_mark:从data_stamp中提取输入序列的时间特征
            seq_x_mark = self.data_stamp[s_begin:s_end]
            #输入序列seq_x,目标序列seq_y，输入序列的时间特征seq_x_mark
            return seq_x, seq_y, seq_x_mark
        
        else:
            #预测模式
            s_begin = index * self.stride
            s_end = s_begin + self.seq_len
            #从 data_x 中提取输入序列
            seq_x = self.data_x[s_begin:s_end]
            #从 data_stamp 中提取输入序列的时间特征
            seq_x_mark = self.data_stamp[s_begin:s_end]
            return seq_x, seq_x_mark

    def __len__(self):
        #获取数据集的长度
        if self.mode != 'pred':#非预测模式
            #数据集的总长度是 len(self.data_x),可用于样本划分的起始点数量是 
            #可用窗口起点数：len(self.data_x) - self.seq_len - self.pred_len + 1
            #如果步长（stride）大于 1，则样本是非重叠提取的，样本总数需除以 stride
            return (len(self.data_x) - self.seq_len - self.pred_len + 1) // self.stride
        else:
            #每个样本仅需要 seq_len 长度的数据，可用窗口起点数是：len(self.data_x) - self.seq_len + 1
            return (len(self.data_x) - self.seq_len + 1) // self.stride


class Exp_Custom(Exp_Basic):
    def __init__(self, args):
        super(Exp_Custom, self).__init__(args)

    def _build_model(self):
        #根据传入的模型名称和配置参数，构建一个模型实例并设置为浮点类型。
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _acquire_device(self):
        #检查配置参数 use_gpu 是否为 True
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
            #创建一个 PyTorch 的 device 对象，指定设备为 GPU
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, data, mode, stride=1):
        if mode == 'train':
            #seq_len：序列长度，表示每个样本的输入数据长度，pred_len：预测长度，表示目标数据的长度，args.freq：时间频率
            dataset = Dataset_Custom(data, self.args.seq_len, self.args.pred_len, freq=self.args.freq, mode='train',
                                     stride=1)
            shuffle = True
        elif mode == 'test':
            dataset = Dataset_Custom(data, self.args.seq_len, self.args.pred_len, freq=self.args.freq, mode='test',
                                     stride=1)
            #设置 shuffle = False，预测时不打乱数据。
            shuffle = False
        elif mode == 'pred':
            dataset = Dataset_Custom(data, self.args.seq_len, self.args.pred_len, freq=self.args.freq, mode='pred',
                                     stride=stride)
            shuffle = False
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers)
        return dataset, dataloader

    def _select_optimizer(self):
        #选择优化器 self.model.parameters()：获取模型中的所有可训练参数  lr=self.args.learning_rate：指定学习率
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        #选择损失函数 选择 均方误差损失函数 (MSE)
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        #初始化 total_loss，用于累积和计算验证集的平均损失
        total_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(vali_loader):
                #batch_x：输入特征，转换为浮点型并转移到设备（CPU/GPU）。
                #batch_y：目标特征，转换为浮点型。
                #batch_x_mark：输入特征的时间戳，转换为浮点型并转移到设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)

                # 输入 batch_x 和 batch_x_mark（时间戳特征）到模型。后两个参数 None 表示不使用某些额外的输入
                outputs = self.model(batch_x, batch_x_mark, None, None)
                #如果 self.args.features 为 'MS'，表示多变量输入，取最后一个特征维度
                f_dim = -1 if self.args.features == 'MS' else 0
                #从预测结果中取最后 pred_len 时间步的预测值。
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                #从真实目标值中取最后 pred_len 时间步的目标值。
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                #计算预测值 outputs 和真实值 batch_y 之间的误差（损失）
                loss = criterion(outputs, batch_y)
                #使用 total_loss.update 累积损失值，按批次记录损失总和和样本数量
                total_loss.update(loss.item(), batch_x.size(0))
        total_loss = total_loss.avg
        self.model.train()
        return total_loss

    def train(self, setting, train_data, vali_data=None, test_data=None):
        #调用 self._get_data 方法，将输入数据（训练、验证、测试）封装成 Dataset 和 DataLoader
        train_data, train_loader = self._get_data(train_data, mode='train')
        if vali_data is not None:
            vali_data, vali_loader = self._get_data(vali_data, mode='test')
        if test_data is not None:
            test_data, test_loader = self._get_data(test_data, mode='test')
        #生成用于保存检查点（模型权重文件）的路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        #记录当前时间，用于计算训练速度和剩余时间
        time_now = time.time()
        #训练集的总批次数
        train_steps = len(train_loader)
        #初始化早停策略 patience 控制容忍的自大连续验证损失未降低的次数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        #选择优化器
        model_optim = self._select_optimizer()
        #选择损失函数
        criterion = self._select_criterion()
        #逐个训练轮次
        for epoch in range(self.args.train_epochs):
            #统计当前 epoch 的迭代次数
            iter_count = 0
            #记录当前 epoch 的训练损失
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            #遍历训练数据集的每个批次
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(train_loader):
                iter_count += 1
                #清空上一次计算的梯度，防止累计
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                #输入 batch_x 和 batch_x_mark（时间戳特征）到模型。后两个参数 None 表示不使用某些额外的输入

                outputs = self.model(batch_x, batch_x_mark, None, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                #计算预测值 outputs 和真实值 batch_y 之间的误差（损失）
                loss = criterion(outputs, batch_y)

                #每训练 100 次迭代，记录并输出当前损失、速度、以及估算的剩余训练时间
                if (i + 1) % 100 == 0:
                    loss_float = loss.item()
                    train_loss.append(loss_float)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_float))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                #反向传播计算梯度
                loss.backward()
                #参数更新
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = None
            test_loss = None
            #如果有验证集，计算验证集损失
            if vali_data is not None:
                vali_loss = self.vali(vali_loader, criterion)
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            #如果有测试集 计算并记录测试集损失
            if test_data is not None:
                test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if early_stopping.early_stop:
                print("Early stopping")
                break
            #调整学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        #加载早停机制保存的最佳模型权重
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        #清理模型文件
        if not self.args.save_model:
            import shutil
            shutil.rmtree(path)
        #返回训练好的模型实例
        return self.model
  
    def test(self, setting, test_data, stride=1):
        #调用 _get_data 方法，生成用于测试的 Dataset 和 DataLoader
        test_data, test_loader = self._get_data(test_data, mode='test', stride=stride)
        #根据传入的 setting 参数，确定模型检查点文件的位置
        model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        print(f'loading model from {model_path}')
        #使用 torch.load 加载保存的权重文件，并通过 load_state_dict 更新当前模型的参数
        self.model.load_state_dict(torch.load(model_path))

        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        #AverageMeter 用于记录和计算平均值（avg）及累计样本量
        mse = AverageMeter()
        mae = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                #输入 batch_x 和 batch_x_mark（时间戳特征）到模型。后两个参数 None 表示不使用某些额外的输入
                outputs = self.model(batch_x, batch_x_mark, None, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                #分别计算 mse和mae
                mse.update(mse_loss(outputs, batch_y).item(), batch_x.size(0))
                mae.update(mae_loss(outputs, batch_y).item(), batch_x.size(0))
        #计算所有测试样本的平均误差值
        mse = mse.avg
        mae = mae.avg
        print('mse:{}, mae:{}'.format(mse, mae))
        return

    def predict(self, setting, pred_data, stride=1):
        pred_data, pred_loader = self._get_data(pred_data, mode='pred', stride=stride)
        model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        print(f'loading model from {model_path}')
        self.model.load_state_dict(torch.load(model_path))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                
                #使用模型对输入数据 batch_x 和时间特征 batch_x_mark 进行预测
                outputs = self.model(batch_x, batch_x_mark, None, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                #将当前批次的预测结果转移到 CPU，并转换为 NumPy 数组
                preds.append(outputs.cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        return pred
