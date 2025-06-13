import torch
import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        # print("dropout:", dropout)# 0.0
        #c_in: 输入特征的维度，即时间序列的特征数量。
        #d_model: 嵌入后的特征维度，即每个特征的维度。
        #dropout: 随机失活的比例，用于防止过拟合。
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.conv3d= nn.Conv3d(in_channels=12, out_channels=12, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.pool3d_layer = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
    def forward(self, x, x_mark):
        #[batch, seq_len, c_in]
        # x=x.reshape(x.shape[0], x.shape[1],2,481,480)
        # x= self.conv3d(x)
        # x= self.pool3d_layer(x)
        # x = self.conv3d(x)
        # x = self.pool3d_layer(x)
        # x = self.conv3d(x)
        # x = self.pool3d_layer(x)
        # x=x.reshape(x.shape[0], x.shape[1],-1)
        x = x.permute(0, 2, 1)

        # print("DataEmbedding_inverted_x.shape",x.shape)#torch.Size([32, 441, 96])
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            # print("x_mark.shape", x_mark.shape)#([16, 12, 3])
            # print("x.shape", x.shape)#[16, 461760, 12]
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        # print("DataEmbedding_inverted_x.shape",x.shape)#torch.Size[16, 461763, 512]
        return self.dropout(x)
