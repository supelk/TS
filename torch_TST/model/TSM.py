# 作者: ZY
# @Time:2025/3/3 12:05
# way down we go
from torch_TST.model.TST import PatchTST_backbone
import torch
import torch.nn as nn
from typing import List
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        # 输入层 -> 第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.layers = nn.Sequential(*layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layers[0](x))  # 第一层需要激活函数
        for layer in self.layers[1:-1]:  # 中间层
            x = self.relu(layer(x))
        # 输出层不需要激活函数（根据任务类型可调整，如分类用Sigmoid）
        x = self.layers[-1](x)
        return x

def test_mlp(id,hd,od):
    x = torch.rand(16,id)
    m = MLP(id,hd,od)
    y = m(x)
    print("x.shape:",x.shape)
    print("y.shape:",y.shape)

class TSM(nn.Module):
    """
    INPUT (bs,c_in,input_len)
    return (bs,output_len,output_dim)
    """
    def __init__(self,c_in,input_len,output_len,MLP_hd: List[int],output_dim):
        super(TSM, self).__init__()
        self.TS = PatchTST_backbone(c_in=c_in, context_window=input_len, target_window=output_len, patch_len=output_len,stride=8)
        self.MLP = MLP(input_dim=c_in,hidden_dims=MLP_hd,output_dim=output_dim)

    def forward(self,x):
        x = x.permute(0, 2, 1)
        x = self.TS(x)
        x = x.permute(0, 2, 1)
        x = self.MLP(x)
        return x

def test_TSM(bs,c_in,input_len,output_len):
    model = TSM(c_in=c_in,input_len=input_len,output_len=output_len,MLP_hd=[32,128,64,8],output_dim=1)
    x = torch.rand(bs, input_len, c_in)
    y = model(x)
    print("x.shape:", x.shape)
    print("output.shape:", y.shape)

# test_TSM(bs=16,c_in=12,input_len=168,output_len=24)