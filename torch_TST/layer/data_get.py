# 作者: ZY
# @Time:2024/12/26 19:48
# way down we go
__all__ = ['give_me_dataloader','give_me_wavelet_dataloader','prepare_data','TSDataset_wavelet','TSDataset']
import numpy as np
import torch
from torch_TST.layer.utils import wavelet_denoise_high_freq
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import r2_score
def give_me_dataloader(df,input_len,output_len):
    train_len = int(df.shape[0]*0.8)
    test_len = df.shape[0]-train_len
    train_df= df[:train_len]
    test_df = df[-test_len:]
    train_set = TSDataset(train_df, input_len=input_len, output_len=output_len)
    test_set = TSDataset(test_df, input_len=input_len, output_len=output_len)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
def give_me_wavelet_dataloader(df,input_len,output_len):
    """ df为原数据，内部包含小波降噪"""
    train_len = int(df.shape[0]*0.8)
    test_len = df.shape[0]-train_len
    kw_raw = df['kw']
    kw_denoised = wavelet_denoise_high_freq(kw_raw)
    kw_denoised = kw_denoised[:-1]
    for i in range(df.shape[0]):
        df['kw'].iloc[i] = kw_denoised[i]
    train_df= df[:train_len]
    test_df = df[-test_len:]
    train_kw = kw_raw[:train_len]
    test_kw = kw_raw[-test_len:]
    train_set = TSDataset_wavelet(train_df,train_kw,input_len,output_len)
    test_set = TSDataset_wavelet(test_df,test_kw,input_len,output_len)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    return train_loader,test_loader
def prepare_data(data_x,data_y,win_size, target_feature_idx, forecast_horizon=24):
    X, y = [], []
    for i in range(len(data_x) - win_size - forecast_horizon):
        X.append(data_x[i:i + win_size, :])  # 输入特征，过去 win_size 小时的数据
        y.append(data_y[i + win_size:i + win_size + forecast_horizon, target_feature_idx])  # 目标特征（未来 24 小时的功率）
    return np.array(X), np.array(y)

from torch.utils.data import Dataset, DataLoader
class   TSDataset(Dataset):
    """" data to input,output，多to单 """
    def __init__(self, data, input_len, output_len, CI):
        self.x = data
        # self.revin_layer = RevIN(CI,)
        self.input_len = input_len
        self.output_len = output_len
        # self.data = self.revin_layer(data,'norm')
        self.y = self.x['kw']
        # self.tr_series = data[['P']]

    def __len__(self):
        data_len = len(self.x) - self.input_len - self.output_len + 1
        return data_len

    def __getitem__(self, idx):
        input_data = torch.tensor(self.x.iloc[idx:idx + self.input_len].values,dtype=torch.float32)
        output_data = torch.tensor(self.y.iloc[idx + self.input_len:idx + self.input_len + self.output_len].values,dtype=torch.float32)
        return input_data, output_data.unsqueeze(-1)
class   TSDataset_wavelet(Dataset):
    """"input,labels to input,output,多to单"""
    def __init__(self, data,labels,input_len, output_len,):
        self.x = data
        self.input_len = input_len
        self.output_len = output_len
        self.y = labels

    def __len__(self):
        data_len = len(self.x) - self.input_len - self.output_len + 1
        return data_len

    def __getitem__(self, idx):
        input_data = torch.tensor(self.x.iloc[idx:idx + self.input_len].values,dtype=torch.float32)
        output_data = torch.tensor(self.y.iloc[idx + self.input_len:idx + self.input_len + self.output_len].values,dtype=torch.float32)
        return input_data, output_data.unsqueeze(-1)
