import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer

#STar Aggregate-Redistribute 模块:对输入时间序列数据进行变换和重组
class STAR(nn.Module):
    #模块初始化
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module，d_series：输入序列的维度，d_core:模块的核心维度，用于内部处理
        """
        #一个线性变换，保持维度不变，d_series->d_series
        self.gen1 = nn.Linear(d_series, d_series)
        #一个线性变换，用于降维到核心维度（d_series -> d_core）。
        self.gen2 = nn.Linear(d_series, d_core)
        #用于融合输入和核心信息（d_series + d_core -> d_series）。
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        #最后进一步变换，输出结果仍保持输入维度（d_series -> d_series）
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        #一个三维张量，形状为 [batch_size, channels, d_series]，通常表示多个时间序列或特征。
        # print("input",input.shape)#[32,442,512]
        batch_size, channels, d_series = input.shape
        #print("batch_size, channels, d_series",batch_size, channels, d_series)#[32,11,512]
        # set FFN Feed-Forward Network
        #将输入通过 gen1 和激活函数 GELU 进行初步非线性变换，得到 combined_mean
        combined_mean = F.gelu(self.gen1(input))
        #print("combined_mean",combined_mean.shape)#
        #再通过 gen2 将数据降维到 d_core
        combined_mean = self.gen2(combined_mean)
        #print("combined_mean降维",combined_mean.shape)#[32, 11, 512]

        # stochastic pooling 随机池化
        #训练阶段
        if self.training:
            #对 combined_mean 进行 softmax 操作，按通道（dim=1）生成概率权重
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            #将概率权重调整形状为 [batch_size * d_core, channels]，为多项式采样做准备。
            ratio = ratio.reshape(-1, channels)
            #使用 torch.multinomial 采样，从每个通道选取一个核心特征的索引
            indices = torch.multinomial(ratio, 1)#[32*512,1]
            #将索引调整形状为[32,512,1]  [32,1,512]
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            #根据采样的索引提取 combined_mean 中的特定值  [32,1,512]
            combined_mean = torch.gather(combined_mean, 1, indices)
            #将这些特征复制并扩展回每个通道。[32,11,512]
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            #计算权重softmax
            weight = F.softmax(combined_mean, dim=1)
            #使用加权求和，计算所有通道的核心特征，扩展回输入的通道数
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        # mlp fusion  特征融合
        #将原始输入和生成的核心特征 combined_mean 拼接（沿最后一个维度，-1），融合原始与核心信息。
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        #通过 gen3 进一步非线性变换，将拼接后的维度投射回 d_series。
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        #再通过 gen4 进一步变换，形成最终的输出。
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat
        #print("output",output.shape)#[32,11,512]
        #output: 经过模块变换后的张量，形状与输入相同。
        return output, None


class Model(nn.Module):
    #初始化函数，嵌入层、编码器和投影层
    def __init__(self, configs):
        super(Model, self).__init__()
        #输入序列的长度
        self.seq_len = configs["seq_len"]
        #输出预测序列的长度
        self.pred_len = configs["pred_len"]
        # Embedding 数据嵌入层
        #DataEmbedding_inverted 是一个嵌入层，将原始输入数据（x_enc）编码为 d_model 维度的高维向量。
        self.enc_embedding = DataEmbedding_inverted(configs['seq_len'], configs['d_model'], configs['dropout'])
        #是否使用归一化（Normalization），用于消除时间序列的非平稳性，增加模型的泛化能力
        self.use_norm = configs["use_norm"]
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAR(configs["d_model"], configs["d_core"]),
                    configs["d_model"],
                    configs["d_ff"],
                    dropout=configs['dropout'],
                    activation=configs['activation'],
                ) for l in range(configs['e_layers'])
            ],
        )        #print("encoder",self.encoder)

        # Decoder 解码器
        #将编码器的输出投影为目标长度的的预测序列
        #输出: [batch_size, pred_len, d_model] 的预测序列。
        self.projection = nn.Linear(configs['d_model'], 12, bias=True)
        self.lin= nn.Linear(7659, 1, bias=True)

    #预测函数执行时间序列预测，包括前处理、编码、解码和后处理
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            #去均值: 减去时间序列的均值，使其以零为中心
            #除以标准差: 对序列进行标准化，减小特征的数值范围。
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        # print("x_enc",x_enc.shape)#[16, 12, 461760]
        _, _, N = x_enc.shape
        # print("x_enc",x_enc.shape)#torch.Size([32, 96, 441])
        # print("x_mark_enc",x_mark_enc.shape)#torch.Size([32, 96, 1])
        #嵌入:将输入时间序列 x_enc 和时间标记 x_mark_enc 转换为高维表示
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # print("enc_out",enc_out.shape)#[16, 461763, 512]
        #编码:使用 self.encoder 提取时间序列的全局和局部模式。
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # print("enc_out",enc_out.shape)#[16, 3039, 64]    Ettm1数据集：[32,11,512]
        #编码器的输出通过线性变换解码为预测值,permute 改变张量的维度顺序，只取长度为N的部分
        dec_out = self.projection(enc_out).permute(0, 2, 1)#b,s,f
        # print("dec_out",dec_out.shape)#torch.Size([16, 12, 7659])

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            #将标准化的数据反变换回原始尺度
            #乘以标准差: 恢复原始数据的幅度
            #加上均值: 还原原始数据
            dec_out = dec_out * stdev
            dec_out = dec_out + means
        dec_out = self.lin(dec_out)
        # print("dec_out_lin",dec_out.shape)#[16,12,12]
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        #print("x_enc",x_enc.shape)#[32,96,7]
       # print("x_mark_enc",x_mark_enc.shape)#[32,96,4]
        #调用 forecast 方法完成预测。只返回预测序列的最后 pred_len 时间步的数据。
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # print("dec_out",dec_out.shape)#[32,96,441]]
        # print("dec_out[:, -self.pred_len:, :]",dec_out[:, -self.pred_len:, :].shape)#dec_out[:, -self.pred_len:, :] torch.Size([32, 96, 441])
        # print("dec_out[:, -self.pred_len:, :]",dec_out[:, -self.pred_len:, :].shape)
        #实际上是获取 dec_out 的最后 pred_len 个时间步的输出，也就是你想要预测的未来序列部分
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
