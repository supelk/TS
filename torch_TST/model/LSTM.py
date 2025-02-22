import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(
            self,
            input_size=12,  # 输入特征维度（对应 features:12）
            lstm_hidden=64,  # LSTM隐藏层维度
            output_len=24,  # 输出时间步（对应 output_len:24）
            num_layers=2
    ):
        super().__init__()
        # LSTM层：提取时序特征
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            batch_first=True,
            num_layers=num_layers,
        )

        # 卷积层：将时间维度从168压缩到24，同时输出1个特征
        self.conv = nn.Conv1d(
            in_channels=lstm_hidden,
            out_channels=1,  # 输出特征数（对应 features:1）
            kernel_size=168 - output_len + 1  # 计算卷积核大小：168-24+1=145
        )

        # 激活函数（可选）
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入形状: (bs, 168, 12)

        # LSTM处理
        lstm_out, _ = self.lstm(x)  # 输出形状: (bs, 168, lstm_hidden)

        # 调整维度以适应Conv1d输入格式 (bs, channels, length)
        lstm_out = lstm_out.permute(0, 2, 1)  # 形状: (bs, lstm_hidden, 168)

        # 卷积压缩时间维度
        conv_out = self.conv(lstm_out)  # 输出形状: (bs, 1, 24)

        # 调整维度为 (bs, 24, 1)
        output = conv_out.permute(0, 2, 1)

        return output  # 形状: (bs, 24, 1)