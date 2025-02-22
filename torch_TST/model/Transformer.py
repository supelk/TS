import torch
import torch.nn as nn
import math


class TF(nn.Module):
    def __init__(
            self,
            input_size=12,  # 输入特征维度
            d_model=64,  # Transformer 特征维度
            nhead=4,  # 注意力头数
            num_layers=2,  # Transformer 编码器层数
            output_len=24  # 输出时间步
    ):
        super().__init__()

        # 输入嵌入层（将原始特征映射到 d_model 维度）
        self.embed = nn.Linear(input_size, d_model)

        # 位置编码（可学习参数）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 168, d_model)  # 输入长度固定为168
        )

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 输出卷积层（压缩时间维度）
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=1,
            kernel_size=168 - output_len + 1  # 168-24+1=145
        )

    def forward(self, x):
        # 输入形状: (bs, 168, 12)

        # 特征嵌入 + 位置编码
        x = self.embed(x)  # (bs, 168, d_model)
        x = x + self.pos_embed  # 添加位置编码

        # Transformer 编码
        x = self.transformer(x)  # (bs, 168, d_model)

        # 调整维度以适应 Conv1d
        x = x.permute(0, 2, 1)  # (bs, d_model, 168)

        # 卷积压缩时间维度
        x = self.conv(x)  # (bs, 1, 24)

        # 调整输出维度
        x = x.permute(0, 2, 1)  # (bs, 24, 1)
        return x

