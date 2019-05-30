#import os
import time
#import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
#import matplotlib.pyplot as plt
#import seaborn as sns
#warnings.filterwarnings("ignore")

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

test_x1 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/test_x1.csv')
test_x2 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/test_x2.csv')
test_x3 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/test_x3.csv')
test_x4 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/test_x4.csv')
test_x5 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/test_x5.csv')
test_x6 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/test_x6.csv')

df_train = [train_x1, train_x2, train_x3, train_x4, train_x5, train_x6, train_x7]
df_train_y = [y1, y2, y3, y4, y5, y6, y7]
df_test = [test_x1, test_x2, test_x3, test_x4, test_x5, test_x6]

train_x_all = pd.concat(df_train)
train_y_all = pd.concat(df_train_y)
test_all = pd.concat(df_test)

train_x_all.drop(labels=['Unnamed: 0','seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
train_y_all.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
test_all = test_all.rename(columns = {'Unnamed: 0':'seg_id'})
test_all = test_all.set_index('seg_id')

"""
scaler data
"""
scaler = StandardScaler()
scaler.fit(train_x_all)
scaled_train_X = pd.DataFrame(scaler.transform(train_x_all), columns=train_x_all.columns)
scaled_test_X = pd.DataFrame(scaler.transform(test_all), columns=test_all.columns)

"""
use pearsonr method to drop un-related column
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
#df = df.loc[df['pval'] < 0.05]
#df[df['col'] == "FFT_Mag_10q15000"]
#df = df[(df.pval < 0.05) & (abs(df.cor)>=0.5)]
#df = df[(df.pval < 0.05)]
df = df[(abs(df.cor)>=0.5)]
#df = df[(abs(df.cor)>=0.55)]
#df = df[(abs(df.cor)>=0.6)]
#df = df[(abs(df.cor)>=0.65)]
#df = df[(abs(df.cor)>=0.7)]

drop_cols = []
for col in scaled_train_X.columns:
    if col not in df['col'].tolist():
        drop_cols.append(col)

scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

"""
build model
"""
params = {#'num_leaves': 21,
         'objective':'regression',
         #'min_data_in_leaf': 88,
         #'max_depth': 7,
         #'learning_rate': 0.001,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         #'max_bin':128,
         "metric": 'mae',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": -1,
         "random_state": 21,
         "verbose":-1}

result = list()
count = 0
for learning_rate in [0.01, 0.001, 0.0001]:
    for max_depth in [5, 7, 9]:
        for num_leaves in [21]:
            for min_data_in_leaf in [21]:
                for max_bin in [255]:
                    count += 1
                    params['learning_rate'] = learning_rate
                    params['max_depth'] = max_depth
                    params['num_leaves'] = num_leaves
                    params['min_data_in_leaf'] = min_data_in_leaf
                    params['max_bin'] = max_bin

                    tune = dict()
                    tune['learning_rate'] = learning_rate
                    tune['max_depth'] = max_depth
                    tune['num_leaves'] = num_leaves
                    tune['min_data_in_leaf'] = min_data_in_leaf
                    tune['max_bin'] = max_bin

                    print ('learning_rate: ',learning_rate)
                    print('max_depth: ', max_depth)
                    print('num_leaves: ', num_leaves)
                    print('min_data_in_leaf: ', min_data_in_leaf)
                    print('max_bin: ', max_bin)

                    X_train, X_test, y_train, y_test = train_test_split(scaled_train_X, train_y_all,
                                                                        test_size=0.3,
                                                                        random_state=21)
                    n_fold = 6
                    folds = KFold(n_splits=n_fold, shuffle=False, random_state=21)
                    train_columns = scaled_train_X.columns.values
                    best_iteration_all = list()
                    test_mae_all = list()
                    oof = np.zeros(len(X_train))
                    test_predictions = np.zeros(len(X_test))
                    predictions = np.zeros(len(scaled_test_X))
                    start_time = time.time()

                    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train.values)):
                        strLog = "fold {}".format(fold_)
                        print(strLog)

                        X_tr, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
                        y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

                        model = lgb.LGBMRegressor(**params, n_estimators=20000, n_jobs=-1)
                        model.fit(X_tr,
                                  y_tr,
                                  eval_set=[(X_tr, y_tr), (X_val, y_val)],
                                  eval_metric='mae',
                                  verbose=10000,
                                  early_stopping_rounds=50)
                        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)

                        predictions += model.predict(scaled_test_X, num_iteration=model.best_iteration_) / folds.n_splits
                        test_predictions += model.predict(X_test, num_iteration=model.best_iteration_) / folds.n_splits
                        best_iteration_all.append(model.best_iteration_)
                        test_mae = mean_absolute_error(test_predictions, y_test)
                        test_mae_all.append(test_mae)
                        print ('Test Dataset MAE: ', str(test_mae))

                    tune['mae'] = test_mae_all[-1]
                    result.append(tune)
                    print ('all mae: ', test_mae_all)
                    end_time = time.time()
                    print(result)
                    submission = pd.read_csv('./dataset/sample_submission.csv', index_col='seg_id')
                    submission.time_to_failure = predictions
                    submission.to_csv('./output/submission_' + str(count) + '.csv', index=True)
                    print('totol run ', str(end_time - start_time))
print (result)
# submission = pd.read_csv('./dataset/sample_submission.csv', index_col='seg_id')
# submission.time_to_failure = predictions
# submission.to_csv('./output/submission'+'_'+fold_+'.csv',index=True)