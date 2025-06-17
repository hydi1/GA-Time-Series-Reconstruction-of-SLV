import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from core_qnn.quaternion_layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.device= device
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        # print("hidden_dim1", self.hidden_dim)  # 64
        self.bias = bias
        self.dtype = dtype
        # 将输入张量和隐藏状态拼接在一起，并通过卷积层计算，可以得到更新门和重置门的值
        # conv_gates = nn.Conv2d(68, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_gates = QuaternionConv(in_channels=input_dim + hidden_dim,#3+5=8
                                         out_channels=2 * self.hidden_dim,#2*5=10
                                         kernel_size=kernel_size,
                                         stride=1,
                                         padding=self.padding,

                                         )

        # self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
        #                             out_channels=2 * self.hidden_dim,
        #                             # for update_gate,reset_gate respectively GRU需要计算更新门和重置门
        #                             kernel_size=kernel_size,
        #                             padding=self.padding,
        #                             bias=self.bias)
        # print("conv_gates",self.conv_gates)  #
        # 计算候选神经记忆，卷积层的输出是将输入张量和当前隐藏状态在通道上拼接后的结果
        #conv_can Conv2d(68, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_can = QuaternionConv(in_channels=input_dim + hidden_dim,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  stride=1)
        # print("conv_can",self.conv_can)

    # 定义一个init_hidden函数，用于初始化隐藏状态，这里使用全0初始化，返回一个形状是（batch_size，hidden_dim,height,width）
    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """

        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        # 将输入张量和当前隐藏状态在通道维度上拼接
        #combined1 torch.Size([12, 68, 10, 10])
        # print("input_tensor", input_tensor.shape)
        # print("h_cur", h_cur.shape)
        self.to(device)
        input_tensor = input_tensor.to(self.device)
        h_cur = h_cur.to(self.device)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        # print("combined1", combined.shape)

        # 通过卷积层计算更新门和重置门
        combined_conv = self.conv_gates(combined)
        #combined_conv1 torch.Size([12, 128, 10, 10])
        # print("combined_conv1", combined_conv.shape)
        # 将卷积层的输出在通道维度上拆分成两个部分，每个子张量的通道数为hidden_dim，分别对应更新门和重置门
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        # 使用sigmoid函数将更新门和重置门限制在0到1之间
        reset_gate = torch.sigmoid(gamma)
        # print("reset_gate1", reset_gate.shape) #reset_gate1 torch.Size([12, 64, 10, 10])
        update_gate = torch.sigmoid(beta)
        # print("update_gate1", update_gate.shape) #update_gate1 torch.Size([12, 64, 10, 10])

        # 将输入张量和重置门与当前隐藏状态的乘积在通道维度上拼接，对应于公式中的[xt,rt*ht-1]
        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        # print("combined2", combined.shape)  #combined2 torch.Size([12, 68, 10, 10])
        # 通过卷积层计算候选神经记忆，对应于公式中的W*concat[xt,rt*ht-1]
        cc_cnm = self.conv_can(combined)
        # print("cc_cnm1", cc_cnm.shape)#cc_cnm1 torch.Size([12, 64, 10, 10])
        # 使用tanh函数将候选神经记忆限制在-1到1之间
        cnm = torch.tanh(cc_cnm)
        # print("cnm1", cnm.shape) #cnm1 torch.Size([12, 64, 10, 10])
        # 根据更新门和候选神经记忆计算下一个隐藏状态，对应于公式中的ht=(1-zt)*ht-1+zt*cnmt
        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        # print("h_next1", h_next.shape) #h_next1 torch.Size([12, 64, 10, 10])
        return h_next


class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype, batch_first=True, bias=True, return_all_layers=False):
        """

        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool 是否将batch放在第一个维度
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        # 调用父类nn.Module的构造函数，初始化ConvGRU
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        # 扩展kernel_size和hidden_dim，使其与num_layers相等
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        # print("kernel_size1", kernel_size)  #[(3,3)]
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        # print("hidden_dim0", hidden_dim)  # [64]
        # 检查kernel_size和hidden_dim的长度是否与num_layers相等
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        self.device = device
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        # print("num_layers", num_layers)
        #为 ConvGRU 网络的每一层创建一个 ConvGRUCell 实例，并将这些实例存储在 cell_list 中，最后将其转换为 ModuleList。
        cell_list = []
        for i in range(0, self.num_layers):
            #如果当前是第0层，输入维度应该是网络的初始输入维度，对于第1层以及之后的层，输入维度cur_intput_dim应该是前一层的隐藏状态的维度
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            # print("cur_input_dim", cur_input_dim)
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        self.to(device)
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            #初始化CONVGRU网络中的每层隐藏状态
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)#时间步长
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]#获取当前层的隐藏状态
            output_inner = []
            for t in range(seq_len):#循环遍历每个时间步 t，seq_len 是输入序列的长度
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function 计算新的隐藏状态
                #cell_list是一个Moduelist,包含了所有层的convgrucell实例，layer_idx是当前层的索引，cur_layer_input[:, t, :, :, :]是当前时间步的输入，h是当前层的隐藏状态
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],  #h= (b,c,h,w)
                                              h_cur=h)
                # print("h",h.shape)#[3072, 5, 3, 107]
                output_inner.append(h)#最后一个循环的时候是output_inner 6 torch.Size([3072, 5, 3, 107])
                # print("output_inner",len(output_inner),output_inner[0].shape)
            # 将输出堆叠起来，形成一个序列
            layer_output = torch.stack(output_inner, dim=1)#  layer_output 的形状应该是 (batch_size, seq_len, 3072, 5, 3, 107)
            # print("layer_output",len(layer_output),layer_output[0].shape)#Last layer output shape: 3072 torch.Size([6, 5, 3, 107])
            # 将当前层的输出作为下一层的输入
            cur_layer_input = layer_output#(batch_size, seq_len, hidden_dim, height, width)

            layer_output_list.append(layer_output)  #layer_output_list 的长度是 num_layers，每一层的输出形状是 (batch_size, seq_len, hidden_dim, height, width)
            last_state_list.append([h])  #  num_layers 个元素，每个元素都是一个形状为 [(3072, 5, 3, 107)] 的列表,(batch_size, hidden_dim, height, width)最后一个时间步的隐藏状态

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]  # (batch_size, seq_len, hidden_dim, height, width)
            last_state_list = last_state_list[-1:]  # 最后一层的的隐藏状态 (batch_size, hidden_dim, height, width)
            # print("Last state list shape:", len(last_state_list[0]), last_state_list[0][0].shape)
            # print("Last layer output shape:", len(layer_output_list[0]),layer_output_list[0][0].shape)
        # print("Last layer output shape:", layer_output_list[0])
        # print("last_state_list",last_state_list[0][0].shape)
        # return layer_output_list, last_state_list
        return layer_output_list[-1]#返回最后一层的输出


#方法最终返回 init_states 列表，其中包含每一层的初始隐藏状态
    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param