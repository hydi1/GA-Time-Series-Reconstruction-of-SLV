import pandas as pd
import numpy as np
import pickle
import joblib
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
from torch.utils.data import Dataset
from models.GAconvgru import Model  # 根据文件和类的实际名称调整路径
import joblib
from utils.timefeatures import time_features
args = {
    'task_name': 'TSUV1',
    'model_id': 'train',
    'model': 'SOFTS',
    'data': 'ssta',
    'features': 'MS',
    'learning_rate': 0.0001,
    'seq_len': 12,
    'label_len': 12,
    'pred_len': 12,
    'd_model': 64,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 256,
    'factor': 1,
    'embed': 'timeF',
    'distil': True,
    'dropout': 0.0,
    'activation': 'gelu',
    'use_gpu': True,
    'train_epochs': 128,
    'batch_size': 16,
    'patience': 128,
    # 'loss': 'MSE',
    "use_norm": False,
    'd_core': 512,
    'freq': 'D',
    'input_size': 1056,
    'hidden_size': 64,
    'output_size': 1,  # 假设是回归任务，输出一个预测值0
    'num_layers': 3,  # 使用3层GRU
    'root_path': r'D:\goole\2025115',
    "data_path": 'TSuv_data_368.npy',
    "target_path": r"D:\goole\GOPRdata\Y非nan -1993_2023.xlsx",
    'target': "OT",  # OT 是什么意思
    'seasonal_patterns': 'Monthly',
    'num_workers': 4,
    'use_amp': False,
    'output_attention': False,
    "lradj": "type1",
    # 'learning_rate': 0.0001,
    'checkpoints': r'D:\project\convgru_TSVU - 反归一化\SOFTS-main\checkpoints',
    "save_model": True,
    'device_ids': [0],
    'scale': True,
    'num_heads': 4,

}


# 假设模型初始化
model = Model(args)

# 加载模型参数
state_dict = torch.load(r"D:\project\convgru_TSVU - 反归一化\SOFTS-main\checkpoints\TSUV5_train_SOFTS_ssta_MS_0.0001_12_12_12_64_2_1_256\checkpoint.pth",weights_only=True)

# 移除 'module.' 前缀
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
# print("Before loading, first parameter sample:", next(iter(model.parameters()))[0])  # 打印初始参数
# 加载更新后的 state_dict
model.load_state_dict(state_dict)
# print("After loading, first parameter sample:", next(iter(model.parameters()))[0])  # 打印加载后的参数
# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
class CustomDataset(Dataset):
    def __init__(self):
        self.__loda_data__()

    def __loda_data__(self):
        # 加载 scaler_x.pkl
        self.scaler_x = joblib.load(r"D:\project\convgru_TSVU - 反归一化\SOFTS-main\scaler_x_time.pkl")
        self.scaler_y = joblib.load(r"D:\project\convgru_TSVU - 反归一化\SOFTS-main\scaler_y_time.pkl")
        # 加载372个时间点的X数据
        file_path = r"D:\goole\2025115\GLOBA\combined_result.npy"
        all_x_data = np.load(file_path)
        all_x_data_2d = all_x_data.reshape(-1, 4*29*6*11)
        self.data_x = self.scaler_x.transform(all_x_data_2d)
        # 加载y数据并归一化 y数据是经过预处理之后的数据，13个月平均和去趋势后的数据
        y_path = r"D:\goole\GOPRdata\y.xlsx"
        y_data = pd.read_excel(y_path).iloc[:, 1].values.reshape(-1, 1)
        self.y_data = y_data  # 保留原始数据，包含 NaN
        # 为模型输入填充 NaN
        y_data_filled = np.where(np.isnan(y_data), np.nanmean(y_data), y_data)  # 用均值填充 NaN
        self.y_data_normalized = self.scaler_y.transform(y_data_filled)  # 用 scaler_y_time.pkl 归一化

        # 读取时间数据
        time_path = r"D:\goole\GOPRdata\time_data.xlsx"
        time_data = pd.read_excel(time_path).iloc[:, 0].values
        df_stamp = pd.to_datetime(time_data)
        dates = pd.DatetimeIndex(df_stamp)
        self.data_stamp = time_features(dates, freq='D')
        self.data_stamp = self.data_stamp.transpose(1, 0)

    def __len__(self):
        return len(self.data_x) - args['seq_len'] + 1

    def __getitem__(self, index):
        if index >= len(self.data_stamp):
            raise IndexError(f"Index {index} out of bounds for axis 0 with size {len(self.data_stamp)}")

        seq_x = torch.tensor(self.data_x[index:index + args['seq_len']], dtype=torch.float32)  # 转换为 Tensor
        seq_y = torch.tensor(self.y_data_normalized[index:index + args['seq_len']], dtype=torch.float32)  # 转换为 Tensor
        seq_x_mark = torch.tensor(self.data_stamp[index:index + args['seq_len']], dtype=torch.float32)  # 转换为 Tensor
        seq_y_mark = torch.tensor(self.data_stamp[index:index + args['seq_len']], dtype=torch.float32)  # 转换为 Tensor

        return seq_x, seq_y, seq_x_mark, seq_y_mark


#一个函数，创建数据集实例并返回数据集对象和对应的 DataLoader
def data_provider():

    def collate_fn(batch):
        try:
            seq_x_batch, seq_y_batch, batch_x_mark, batch_y_mark = zip(*batch)

            seq_x_batch = torch.stack(seq_x_batch, dim=0)
            seq_y_batch = torch.stack(seq_y_batch, dim=0)

            batch_x_mark = [
                torch.tensor(np.array(item), dtype=torch.float32) if not isinstance(item, torch.Tensor) else item for
                item in batch_x_mark]
            # batch_x_mark = [torch.tensor(item) if isinstance(item, list) else item for item in batch_x_mark]
            batch_x_mark = torch.stack(batch_x_mark, dim=0)
            # batch_y_mark = [torch.tensor(item) if isinstance(item, list) else item for item in batch_y_mark]
            batch_y_mark = [
                torch.tensor(np.array(item), dtype=torch.float32) if not isinstance(item, torch.Tensor) else item for
                item in batch_y_mark]
            batch_y_mark = torch.stack(batch_y_mark, dim=0)
            return seq_x_batch, seq_y_batch, batch_x_mark, batch_y_mark
        except Exception as e:
            print("Batch processing error:", e)
            raise
#data_set 是一个 CustomDataset 类的实例，包含预加载的数据（self.data_x、self.y_data_normalized、self.data_stamp）
# #data_set 是一个 CustomDataset 实例，封装了你的数据和访问逻辑。它是一个可迭代对象，DataLoader 会通过 __len__ 获取样本数，通过 __getitem__ 获取每个样本。
    data_set=CustomDataset()
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=0,
        drop_last=False, collate_fn=collate_fn)
    #len(data_loader)=len(data_set)//batch_size+1,即451//32+1=15,即每个epoch有15个batch
    return data_set, data_loader

#调用 data_provider()，获取数据集和数据加载器，用于后续推理。test_data可以访问原始数据（test_data.data_x、test_data.y_data_normalized），test_loader实例用于迭代批次数据
test_data, test_loader= data_provider()

#切换到评估模式
model.eval()

# 推理并保存结果
full_sequence = np.zeros((372, 1))  # 存储预测值
full_y_true = np.zeros((372, 1))   # 存储真实值
counts = np.zeros((372, 1))        # 记录覆盖次数

all_outputs = []  # 用于存储所有反归一化预测
full_sequence = np.zeros((372, 1))  # 目标序列：372 个时间步
counts = np.zeros((372, 1))  # 记录每个时间步的覆盖次数
with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        #前向推理
        output = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        print(f"Batch {i+1} model_outputs shape: {output.shape}")
        # 转换为 numpy
        outputs_cpu = output.cpu().numpy()  # (16, 12, 1)
        batch_y_cpu = batch_y.cpu().numpy()  # (16, 12, 1)
        # 反归一化
        outputs_flat = outputs_cpu.reshape(-1, 1)
        batch_y_flat = batch_y_cpu.reshape(-1, 1)
        outputs_fgyh = test_data.scaler_y.inverse_transform(outputs_flat)  # 用 scaler_y_time.pkl 反归一化 output
        batch_y_fgyh = test_data.scaler_y.inverse_transform(batch_y_flat)  # 用 scaler_y_time.pkl 反归一化 batch_y
        # 恢复形状
        outputs_original = outputs_fgyh.reshape(outputs_cpu.shape)  # (16, 12, 1)
        batch_y_original = batch_y_fgyh.reshape(batch_y_cpu.shape)  # (16, 12, 1)

        # 映射到完整时间轴
        batch_size = outputs_original.shape[0]
        start_idx = i * args['batch_size']
        pred_len = args['pred_len']

        for j in range(batch_size):
            pred_start = start_idx + j
            pred_end = pred_start + pred_len
            valid_end = min(pred_end, 372)
            slice_len = valid_end - pred_start

            # 累加预测值和真实值
            full_sequence[pred_start:valid_end] += outputs_original[j, :slice_len, 0].reshape(-1, 1)
            full_y_true[pred_start:valid_end] += batch_y_original[j, :slice_len, 0].reshape(-1, 1)
            counts[pred_start:valid_end] += 1

    # # 计算平均值
    # full_sequence = full_sequence / np.maximum(counts, 1)  # 预测值平均
    # full_y_true = full_y_true / np.maximum(counts, 1)  # 真实值平均
    #
    # # 保存到 DataFrame
    # df = pd.DataFrame({
    #     'True_Values': full_y_true.flatten(),  # 反归一化的真实值
    #     'Predicted_Values': full_sequence.flatten()  # 反归一化的预测值
    # })
    # df.to_excel('reconstructed_372_timesteps_with_true.xlsx', index=False)
    # print("Reconstructed shape:", full_sequence.shape)
    # print("True values shape:", full_y_true.shape)
    # 计算平均值，仅对非 NaN 点
    full_sequence = full_sequence / np.maximum(counts, 1)
    # 恢复原始 y_data 的 NaN
    full_y_true_raw = test_data.y_data  # 直接使用原始 y_data，包含 NaN
    full_y_true = full_y_true / np.maximum(counts, 1)  # 计算平均后的 y_true

    # 保存到 DataFrame
    df = pd.DataFrame({
        'True_Values': full_y_true_raw.flatten(),  # 保留 NaN 的原始真实值
        # 'True_Values_Averaged': full_y_true.flatten(),  # 平均后的真实值
        'Predicted_Values': full_sequence.flatten()
    })
    df.to_excel('reconstructed_372_timesteps_with_true2.xlsx', index=False)

    # 计算 RMSE（仅非 NaN 点）
    mask = ~np.isnan(full_y_true_raw) & ~np.isnan(full_sequence)
    rmse = np.sqrt(np.mean((full_y_true_raw[mask] - full_sequence[mask]) ** 2))
    print(f"RMSE (non-NaN points): {rmse:.4f}")
    print("Reconstructed shape:", full_sequence.shape)
    print("True values shape:", full_y_true_raw.shape)