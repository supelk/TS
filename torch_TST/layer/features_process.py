# 作者: ZY
# @Time:2024/12/26 19:47
# way down we go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import r2_score


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


def normalize_dataframe(train_df, test_df, features):
    scaler = MinMaxScaler()
    scaler.fit(train_df[features])
    train_data = pd.DataFrame(scaler.transform(train_df[features]), columns=features, index=train_df.index)
    test_data = pd.DataFrame(scaler.transform(test_df[features]), columns=features, index=test_df.index)
    return train_data, test_data, scaler


def add_time_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    return df