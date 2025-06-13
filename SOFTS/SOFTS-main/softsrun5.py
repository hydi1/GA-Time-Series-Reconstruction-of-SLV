import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

def train_and_evaluate_model():
    """
    Train and evaluate the SOFTS model using the provided training, validation, and test data.
    """
    # Configure arguments for the experiment
    args = {
        'task_name': 'vst',
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
        'output_size': 1,  # 假设是回归任务，输出一个预测值
        'num_layers': 3,  # 使用3层GRU
        'root_path': r'D:\goole\2025115',
        "data_path": 'TSuv_data_368.npy',
        "target_path": r"D:\goole\GOPRdata\Y非nan -1993_2023.xlsx",
        'target': "OT",  # OT 可能是目标变量名称（如 Ocean Temperature），需确认
        'seasonal_patterns': 'Monthly',
        'num_workers': 4,
        'use_amp': False,
        'output_attention': False,
        "lradj": "type1",
        # 'learning_rate': 0.0001,
        'checkpoints': r'D:\project\SOFTS_TS -反归一化\SOFTS-main\checkpoints',
        "save_model": True,
        'device_ids': [0],
        'scale': True,
        'num_heads': 4,
    }

    # Initialize experiment
    exp = Exp_Long_Term_Forecast(args)
    # Start training
    print(f"Starting training with model ID: {args['model_id']}")
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
    rmse_batch_unnorm_avg = result['Batch Unnormalized: rmse']
    mae_batch_unnorm_avg = result['Batch Unnormalized: mae']
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