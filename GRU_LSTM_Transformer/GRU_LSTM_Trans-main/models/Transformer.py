import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs['seq_len']
        print("self.seq_len", self.seq_len)
        self.pred_len = configs['pred_len']
        self.input_size = configs['input_size']
        self.d_model = configs.get('d_model', 64)  # Transformer模型维度
        self.nhead = configs.get('nhead', 4)       # 注意力头数
        self.num_layers = configs.get('num_layers', 2)  # Transformer层数
        self.output_size = configs['output_size']
        self.dropout = configs.get('dropout', 0.3)

        # 输入投影层
        self.input_projection = nn.Linear(self.input_size, self.d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # 输出全连接层
        self.fc = nn.Linear(self.d_model, self.output_size)

        # Dropout层
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        # x 的 shape 为 (batch_size, seq_len, input_size)

        # 输入投影
        x = self.input_projection(x)  # shape: (batch_size, seq_len, d_model)

        # 添加位置编码
        x = self.pos_encoder(x)

        # Transformer编码器处理
        output = self.transformer_encoder(x)  # shape: (batch_size, seq_len, d_model)

        # 应用dropout
        output = self.dropout(output)

        # 输出投影
        output = self.fc(output)  # shape: (batch_size, seq_len, output_size)
        print('output', output.shape)

        return output