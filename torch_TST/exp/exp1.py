# 作者: ZY
# @Time:2025/1/3 14:26
# way down we go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from layer.RevIN import RevIN
from model.TST import PatchTST_backbone
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

data_features_dict = {
    "ELBSL4-12570S机组系统_L4-125S压缩机组_01#压缩机_压缩机1能级状态": "cpr1",
    "ELBSL4-12570S机组系统_L4-125S压缩机组_02#压缩机_压缩机2能级状态": "cpr2",
    "ELBSL4-12570S机组系统_L4-125S压缩机组_03#压缩机_压缩机3能级状态": "cpr3",
    "ELBSL4-12570S机组系统_L4-70S压缩机组_04#压缩机_压缩机4能级状态": "cpr4",
    "ELBSL4-12570S机组系统_L4冷凝设备组_01#水泵_蒸发冷水泵": "cdr_pump",
    "ELBSL4-12570S机组系统_L4冷凝设备组_01#风机_蒸发冷风机1": "cdr_fan1",
    "ELBSL4-12570S机组系统_L4冷凝设备组_02#风机_蒸发冷风机2": "cdr_fan2",
    "ELBSL4-12570S机组系统_L4冷凝设备组_03#风机_蒸发冷风机3": "cdr_fan3",
    "ELBSL4-12570S机组系统_L4-125S压缩机组_冷冻吸气压力": "in_pa",
    "ELBSL4-12570S机组系统_L4-125S压缩机组_冷冻吸气温度": "in_temp",
    "ELBSL4-12570S机组系统_L4-125S压缩机组_排气压力": "ex_pa",
    "ELBSL4-12570S机组系统_L4-125S压缩机组_供液温度": "liq_temp",
    "ELBSL4-12570S机组系统_L4-125S压缩机组_排气温度": "ex_temp",
    "ELBSL4-12570S机组系统_L4-125S压缩机组_环境温度": "env_temp",
    "3#冷冻库_01#风机_风机状态": "fre3_fan1",
    "冷藏库_01#风机_风机状态": "refr_fan",
    "1#冷冻库_01#风机_风机状态": "fre1_fan1",
    "2#冷冻库_01#风机_风机状态": "fre2_fan1",
    "3#冷冻库_02#风机_风机状态": "fre3_fan2",
    "冷藏库_02#风机_风机状态": "refr_fan2",
    "1#冷冻库_02#风机_风机状态": "fre1_fan2",
    "2#冷冻库_02#风机_风机状态": "fre2_fan2",
    "月台_01#02#风机_风机状态": "pf_fan12",
    "1#冷冻库_03#风机_风机状态": "fre1_fan3",
    "2#冷冻库_03#风机_风机状态": "fre2_fan3",
    "3#冷冻库_03#风机_风机状态": "fre3_fan3",
    "月台_03#04#风机_风机状态": "pf_fan34",
    "3#冷冻库_04#风机_风机状态": "fre3_fan4",
    "1#冷冻库_04#风机_风机状态": "fre1_fan4",
    "2#冷冻库_04#风机_风机状态": "fre2_fan4",
    "1#冷冻库_05#风机_风机状态": "fre1_fan5",
    "月台_05#06#风机_风机状态": "pf_fan56",
    "1#冷冻库_06#风机_风机状态": "fre1_fan6",
    "1#冷冻库_冷冻1_传感器1温度": "fre1_temp1",
    "1#冷冻库_冷冻1_传感器2温度": "fre1_temp2",
    "1#冷冻库_冷冻1_传感器3温度": "fre1_temp3",
    "1#冷冻库_冷冻1_传感器4温度": "fre1_temp4",
    "1#冷冻库_冷冻1_传感器5温度": "fre1_temp5",
    "1#冷冻库_冷冻1_传感器6温度": "fre1_temp6",
    "1#冷冻库_冷冻1_平均温度": "fre1_temp_avg",
    "2#冷冻库_冷冻库2门1_冷冻库2门1": "fre2_door1",
    "2#冷冻库_冷冻库2门2_冷冻库2门2": "fre2_door2",
    "2#冷冻库_冷冻库2门3_冷冻库2门3": "fre2_door3",
    "2#冷冻库_冷冻2_传感器1温度": "fre2_temp1",
    "2#冷冻库_冷冻2_传感器2温度": "fre2_temp2",
    "2#冷冻库_冷冻2_传感器3温度": "fre2_temp3",
    "2#冷冻库_冷冻2_传感器4温度": "fre2_temp4",
    "2#冷冻库_冷冻2_平均温度": "fre2_temp_avg",
    "3#冷冻库_冷冻3_传感器1温度": "fre3_temp1",
    "3#冷冻库_冷冻3_传感器2温度": "fre3_temp2",
    "3#冷冻库_冷冻3_传感器3温度": "fre3_temp3",
    "3#冷冻库_冷冻3_传感器4温度": "fre3_temp4",
    "3#冷冻库_冷冻3_平均温度": "fre3_temp_avg",
    "ELBSL4-12570S机组系统_其他_L4机组电表_L4机组_Psum": "kw",
    "1#冷冻库_冷冻1_温度设定上限": "fre1_temp_up",
    "1#冷冻库_冷冻1_温度设定下限": "fre1_temp_low",
    "2#冷冻库_冷冻2_温度设定上限": "fre2_temp_up",
    "2#冷冻库_冷冻2_温度设定下限": "fre2_temp_low",
    "3#冷冻库_冷冻3_温度设定上限": "fre3_temp_up",
    "3#冷冻库_冷冻3_温度设定下限": "fre3_temp_low"
}
features = [
    'ELBSL4-12570S机组系统_其他_L4机组电表_L4机组_Psum',
    'ELBSL4-12570S机组系统_L4冷凝设备组_01#水泵_蒸发冷水泵',
    'ELBSL4-12570S机组系统_L4冷凝设备组_01#风机_蒸发冷风机1',
    '1#冷冻库_05#风机_风机状态',
    'ELBSL4-12570S机组系统_L4-125S压缩机组_03#压缩机_压缩机3能级状态',
    'ELBSL4-12570S机组系统_L4-125S压缩机组_01#压缩机_压缩机1能级状态',
    '1#冷冻库_03#风机_风机状态',
    'ELBSL4-12570S机组系统_L4-125S压缩机组_02#压缩机_压缩机2能级状态',
]
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
features_code =[]
for feature in features:
    features_code.append(data_features_dict[feature])
df = pd.read_csv(r"E:\Deeplearn\time series\state2\data\data_wh1_1hour.csv", index_col=0, parse_dates=['time'])
def add_time_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    return df
features_code.append('hour')
features_code.append('day_of_week')
features_code.append('day_of_month')
features_code.append('month')
df = add_time_features(df)
label = df['kw'].copy()