import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchinfo import summary
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# 定义计算参数量的函数
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
def train_and_evaluate_model():
    """
    Train and evaluate the SOFTS model using the provided training, validation, and test data.
    """
    # Configure arguments for the experiment
    args = {
        'task_name': 'GAConvgru_TSUV',
        'model_id': 'train',
        'model': 'GAconvgru',
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
        'use_norm': False,
        'd_core': 512,
        'freq': 'D',
        'input_size': 1056,
        'hidden_size': 64,
        'output_size': 1,
        'num_layers': 3,
        'root_path': r'D:\海平面变率\nepo',
        # 'data_path': 'TSuv_data_368.npy',
        'data_path': 'TSuv_data_368_with_curl.npy',
        'target_path': r'D:\海平面变率\nepo\Y非nan -1993_2023.xlsx',
        'target': 'OT',
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        'lradj': 'type1',
        'checkpoints': r'D:\project\convgru_TSVU - 反归一化\SOFTS-main\checkpoints',
        'save_model': True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

    # Initialize experiment
    exp = Exp_Long_Term_Forecast(args)
    # Start training
    print(f"Starting training with model ID: {args['model_id']}")

    # 显式构建模型以访问它
    model = exp._build_model()

    # 计算并打印参数量
    print("总可训练参数量：", count_param(model))

    # 获取一个真实的输入批次
    train_data, train_loader = exp._get_data(flag='train')
    batch = next(iter(train_loader))  # 获取第一个批次
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch

    # 移动到与模型相同的设备
    device = torch.device('cuda' if args['use_gpu'] and torch.cuda.is_available() else 'cpu')
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device) if batch_x_mark is not None else None
    batch_y = batch_y.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device) if batch_y_mark is not None else None

    # 构造 x_dec（根据你的 forward 方法逻辑）
    dec_inp = batch_y  # 假设 x_dec 是 batch_y，或者根据需要调整

    # 使用 input_data 传递实际输入
    input_data = (batch_x, batch_x_mark, dec_inp, batch_y_mark)
    print(summary(model, input_data=input_data))

    exp.train(args)
    print("Training completed!")

    # Evaluate on validation data
    print("Starting evaluation on validation set...")
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args['task_name'],
        args['model_id'],
        args['model'],
        args['data'],
        args['features'],
        args['seq_len'],
        args['label_len'],
        args['pred_len'],
        args['d_model'],
        args['e_layers'],
        args['d_layers'],
        args['d_ff'],
        args['factor'],
        args['embed'],
        args['distil'],
        args['target']
    )
    # 调用 test 方法，并接收所有返回值
    result = exp.test(setting)
    # 假设 test 返回一个字典，包含 rmse_batch_unnorm_avg 和 mae_batch_unnorm_avg
    rmse_batch_unnorm_avg = result['rmse_batch_unnorm_avg']
    mae_batch_unnorm_avg = result['mae_batch_unnorm_avg']
    print("Evaluation completed!")
    return rmse_batch_unnorm_avg, mae_batch_unnorm_avg  # 只返回未反归一化的指标

if __name__ == "__main__":
    # 多次运行并统计结果
    import numpy as np

    rmse_list = []
    mae_list = []
    for _ in range(5):  # 运行 5 次
        rmse, mae = train_and_evaluate_model()
        rmse_list.append(rmse)
        mae_list.append(mae)

    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)
    mean_mae = np.mean(mae_list)
    std_mae = np.std(mae_list)
    print(f"Batch Unnormalized RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}")
    print(f"Batch Unnormalized MAE: {mean_mae:.3f} ± {std_mae:.3f}")
