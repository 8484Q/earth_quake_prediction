import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler


#os.chdir('C:/Users/mengnan.yi/PycharmProjects/kaggle/earth_quake')
print(os.listdir("./"))

train_x1 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x1.csv',dtype=np.float32)
train_x2 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x2.csv',dtype=np.float32)
train_x3 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x3.csv',dtype=np.float32)
train_x4 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x4.csv',dtype=np.float32)
train_x5 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x5.csv',dtype=np.float32)
train_x6 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x6.csv',dtype=np.float32)
train_x7 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x7.csv',dtype=np.float32)

y1 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y1.csv')
y2 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y2.csv')
y3 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y3.csv')
y4 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y4.csv')
y5 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y5.csv')
y6 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y6.csv')
y7 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y7.csv')


df_train = [train_x1, train_x2, train_x3, train_x4, train_x5, train_x6, train_x7]
df_train_y = [y1, y2, y3, y4, y5, y6, y7]

train_x_all = pd.concat(df_train)
train_y_all = pd.concat(df_train_y)

train_x_all.drop(labels=['Unnamed: 0','seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
train_y_all.drop(labels=['Unnamed: 0'], axis=1, inplace=True)

scaler = StandardScaler()
scaler.fit(train_x_all)
scaled_train_X = pd.DataFrame(scaler.transform(train_x_all), columns=train_x_all.columns)

"""
plot relationship between features and target time
"""
def plot_feature_time(x, y):
    fig, ax1 = plt.subplots(figsize=(16, 8))
    plt.title("Plot")
    plt.plot(x, color='r')
    ax1.set_ylabel('acoustic_data', color='r')
    plt.legend(['acoustic_data'])
    ax2 = ax1.twinx()
    plt.plot(y, color='b')
    ax2.set_ylabel('time_to_failure', color='b')
    plt.legend(['time_to_failure'], loc=(0.875, 0.9))
    plt.grid(False)

feature_column = train_x_all['abs_q05_0'].values[::10]
time_to_failure = train_y_all['time_to_failure'].values[::10]
plot_feature_time(feature_column, time_to_failure)

"""
output cor and pval for features
"""
pcol = []
pcor = []
pval = []
y = train_y_all['time_to_failure'].values

for col in scaled_train_X.columns:
    pcol.append(col)
    pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
    pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))

df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
df.sort_values(by=['cor', 'pval'], inplace=True)
df.dropna(inplace=True)
#df = df[(df.pval < 0.05) & (abs(df.cor)>=0.5)]
df = df[(abs(df.cor)>=0.5)]
#df = df[(abs(df.cor)>=0.55)]
#df = df[(abs(df.cor)>=0.6)]
#df = df[(abs(df.cor)>=0.65)]
#df = df[(abs(df.cor)>=0.7)]

df.to_csv('./output/feature_cor.csv')

