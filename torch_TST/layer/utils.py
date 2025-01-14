# 作者: ZY
# @Time:2025/1/5 14:24
# way down we go
__all__ =  ['masked_mape_loss','r2_score','EarlyStopCallback','plot_learning_curves','wavelet_denoise_high_freq','plot_list','show_labelperdict']
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def masked_mape_loss(y_pred, y_true):
    loss = torch.abs(torch.div(y_true - y_pred, y_true)) * 100
    return loss.mean()
def r2_score(y_true, y_pred):
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2
class EarlyStopCallback:
    def __init__(self, patience=5, min_delta=0.01):
        """

        Args:
            patience (int, optional): Number of epochs with no improvement after which training will be stopped.. Defaults to 5.
            min_delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
                change of less than min_delta, will count as no improvement. Defaults to 0.01.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = - np.inf
        self.counter = 0

    def __call__(self, metric):
        if metric >= self.best_metric + self.min_delta:
            # update best metric
            self.best_metric = metric
            # reset counter
            self.counter = 0
        else:
            self.counter += 1

    @property
    def early_stop(self):
        return self.counter >= self.patience
def plot_learning_curves(record_dict, sample_step=500):
    # build DataFrame
    train_df = pd.DataFrame(record_dict["train"]).set_index("step").iloc[::sample_step]
    # val_df = pd.DataFrame(record_dict["val"]).set_index("step")

    # plot
    for idx, item in enumerate(train_df.columns):
        plt.plot(train_df.index, train_df[item], label=f"train_{item}")
        # plt.plot(val_df.index, val_df[item], label=f"val_{item}")
        plt.grid()
        plt.legend()
        # plt.xticks(range(0, train_df.index[-1], 10*sample_step), range(0, train_df.index[-1], 10*sample_step))
        plt.xlabel("step")

        plt.show()


import torch
import torch.nn.functional as F
import pywt

def wavelet_denoise_high_freq(signal, wavelet='db7', level=4, threshold_mode='soft'):
    # 1. 使用小波变换分解信号
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 2. 高频系数阈值处理
    # 高频噪声通常出现在小波分解的较高层次（细节系数）
    thresholded_coeffs = []
    for i, coeff in enumerate(coeffs):
        if i == 0:
            # 低频部分（逼近系数）通常保留
            thresholded_coeffs.append(coeff)
        else:
            # 高频部分（细节系数）进行阈值处理
            threshold = np.median(np.abs(coeff)) / 0.6745
            if threshold_mode == 'soft':
                thresholded_coeffs.append(pywt.threshold(coeff, threshold, mode='soft'))
            elif threshold_mode == 'hard':
                thresholded_coeffs.append(pywt.threshold(coeff, threshold, mode='hard'))
            else:
                raise ValueError("Unsupported threshold mode.")

    # 3. 重建去噪后的信号
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet)
    return denoised_signal

def plot_list(list):
    plt.figure(figsize=(20, 8))
    plt.plot(list)
    plt.show()

def show_labelperdict(label,predict):
    # 绘制图表
    plt.figure(figsize=(20, 8))
    plt.plot(label, label='True Values (label)',color='blue')  # 折线图表示真实值
    plt.plot(predict, label='Predicted Values (predict)', color='red')  # 折线图表示预测值

    # 添加标题和标签
    plt.title('True vs Predicted Values', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Values', fontsize=12)

    # # 添加图例
    plt.legend(fontsize=12)

    # 显示图表
    plt.grid()
    plt.show()
