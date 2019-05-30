import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

train = pd.read_csv('./dataset/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
train['diff']=train['time_to_failure'].diff(1)
scope = pd.Series(train.loc[train['diff']>1.0].index)

test_files = os.listdir("./dataset/test")

"""
plot train 16 peak data
"""
def plot_quake(x,y):
    fig, ax1 = plt.subplots(figsize=(20, 8))
    ax1.plot(x, color='r')
    ax1.set_xlabel("Index")
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(y, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)

for i in scope:
    start_idx = i+1-150000
    end_idx = i+1
    train_ad_sample_df= train['acoustic_data'].values[start_idx:end_idx]
    train_ttf_sample_df = train['time_to_failure'].values[start_idx:end_idx]
    plot_quake(train_ad_sample_df,train_ttf_sample_df)

"""
plot test data
"""
def plot_quake_test(n,x):
    fig, ax1 = plt.subplots(figsize=(20, 8))
    ax1.plot(x.acoustic_data.values, color='r')
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Signal")
    ax1.set_ylim([-100, 100])
    ax1.set_title("Test {}".format(test_files[n]))

for n in range(4):
    seg = pd.read_csv('./dataset/test/'  + test_files[n])
    plot_quake_test(n,seg)

"""
plot train 16 after peak 150000 data
"""
for i in scope:
    start_idx = i
    end_idx = i+150000
    train_ad_sample_df= train['acoustic_data'].values[start_idx:end_idx]
    train_ttf_sample_df = train['time_to_failure'].values[start_idx:end_idx]
    plot_quake(train_ad_sample_df,train_ttf_sample_df)