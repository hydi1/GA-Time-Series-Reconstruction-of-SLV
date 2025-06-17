import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", **kwargs):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        #将维度从 d_model 扩展到 d_ff，使用 1D 卷积
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        #将维度从 d_ff 恢复到 d_model，使用 1D 卷积
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        #LayerNorm 层，用于规范化数据
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        #随机失活层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
        #激活函数
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        new_x, attn = self.attention(
            #输入是x,用作query\key\value,并且支持传入额外的参数 attn_mask\tau\delta
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta,
            **kwargs
        )
        #添加残差连接
        x = x + self.dropout(new_x)
        #第一层归一化
        y = x = self.norm1(x)
        # print("EncoderLayer y:", y.shape)#([16, 28803, 64])
        #转置 从 [batch_size, seq_len, d_model] 到 [batch_size, d_model, seq_len]
        #使用1D卷积将特征维度扩展到 d_ff 应用激活函数、Dropout
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        #使用 conv2 将特征维度从 d_ff 缩减回 d_model transpose 转置回 原始形状[batch_size,seq_len,d_model]
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        #添加残差连接 将前馈神经网络的输出与原始输入相加
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        #多头自注意力模块，多个EncoderLayer的实例
        self.attn_layers = nn.ModuleList(attn_layers)
        #print("Encoder attn_layers:", self.attn_layers)
        #卷积层
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        #print("Encoder conv_layers:", self.conv_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        # x [B, L, D],attn_mask 注意力掩码
        attns = []
        # print('self.conv_layers',self.conv_layers)#none
        if self.conv_layers is not None:

            #使用 enumerate(zip(...)) 遍历 attn_layers 和 conv_layers，同时获取当前的索引 i
            #每次迭代，attn_layer 和 conv_layer 分别是当前的注意力层和卷积层。
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            #处理最后一个注意力层，对当前序列表示 x 应用最后一层注意力机制。
            #将最后一层的注意力权重 attn 也保存到 attns 列表中。
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, **kwargs)
                attns.append(attn)
        # print('self.norm',self.norm)#none
        if self.norm is not None:
            x = self.norm(x)

        return x, attns
