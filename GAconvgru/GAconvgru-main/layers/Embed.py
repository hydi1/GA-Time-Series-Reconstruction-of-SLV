import torch
import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        #c_in: 输入特征的维度，即时间序列的特征数量。
        #d_model: 嵌入后的特征维度，即每个特征的维度。
        #dropout: 随机失活的比例，用于防止过拟合。
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        #[batch, seq_len, c_in]
        x = x.permute(0, 2, 1)

        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            # print("x_mark.shape", x_mark.shape)#([16, 12, 3])
            # print("x.shape", x.shape)#[16, 461760, 12]
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))

        return self.dropout(x)
