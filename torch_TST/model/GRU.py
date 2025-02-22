import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(
            self,
            input_size=12,  # 输入特征维度（对应 features:12）
            hidden_size=64,  # GRU隐藏层维度
            output_len=24,  # 输出时间步（对应 output_len:24）
    ):
        super().__init__()

        # GRU层：提取时序特征
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # 全连接层：将GRU输出映射到目标维度
        self.fc = nn.Linear(hidden_size, 1)

        # 卷积层：将时间维度从168压缩到24
        self.conv = nn.Conv1d(
            in_channels=168,
            out_channels=output_len,
            kernel_size=1
        )

    def forward(self, x):
        # 输入形状: (bs, 168, 12)

        # GRU处理
        gru_out, _ = self.gru(x)  # 输出形状: (bs, 168, hidden_size)

        # 全连接层：映射到1维特征
        fc_out = self.fc(gru_out)  # 输出形状: (bs, 168, 1)

        # 调整维度以适应Conv1d输入格式 (bs, channels, length)
        # fc_out = fc_out.permute(0, 2, 1)  # 形状: (bs, 1, 168)

        # 卷积压缩时间维度
        conv_out = self.conv(fc_out)  # 输出形状: (bs, 24, 1)

        # 调整维度为 (bs, 24, 1)
        output = conv_out

        return output  # 形状: (bs, 24, 1)

# # 初始化模型
# model = GRU(input_size=12, hidden_size=64)
#
# # 生成模拟输入数据
# bs = 32  # 批量大小
# x = torch.randn(bs, 168, 12)  # 输入形状 (32, 168, 12)
#
# # 前向传播
# output = model(x)
#
# # 检查输出形状
# print(output.shape)  # 应输出 torch.Size([32, 24, 1])